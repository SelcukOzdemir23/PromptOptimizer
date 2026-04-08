"""
EvoPrompt Optimizer - API Health Check & Connection Test

Run this script to verify that:
1. The Groq API key is configured
2. The groq SDK can authenticate
3. The target model is accessible
4. A basic text generation call succeeds

Usage:
    source venv/bin/activate
    python tests/test_api_health.py
"""

import sys
import time
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from groq import Groq
from groq import APIError
import config


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check_api_key() -> bool:
    """Verify that an API key is configured."""
    print_section("1. API Key Check")

    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_api_key_here":
        print("❌ FAIL: GROQ_API_KEY is not set.")
        print("   → Get your free API key from:")
        print("     https://console.groq.com/keys")
        return False

    masked = config.GROQ_API_KEY[:4] + "..." + config.GROQ_API_KEY[-4:]
    print(f"✅ PASS: API key is set ({masked})")
    return True


def check_sdk_import() -> bool:
    """Verify that the groq SDK can be imported."""
    print_section("2. SDK Import Check")

    try:
        from groq import Groq
        import groq
        version = getattr(groq, '__version__', 'unknown')
        print(f"✅ PASS: groq SDK imported (version: {version})")
        return True
    except ImportError as e:
        print(f"❌ FAIL: Cannot import groq SDK: {e}")
        return False


def check_model_availability() -> bool:
    """Verify that the configured model is accessible."""
    print_section("3. Model Availability Check")

    client = Groq(api_key=config.GROQ_API_KEY)
    print(f"   Target model: {config.GROQ_MODEL}")

    try:
        # Groq doesn't have a models.list() in free tier, so we test
        # with a minimal request instead
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model=config.GROQ_MODEL,
            max_tokens=5,
        )
        message = response.choices[0].message
        text = message.content if message.content else "(empty)"
        print(f"✅ PASS: {config.GROQ_MODEL} responded: '{text[:50]}'")
        return True

    except APIError as e:
        print(f"❌ FAIL: API error: {e}")
        status_code = getattr(e, 'status_code', None)  # type: ignore[arg-type]
        if status_code == 401:
            print("   → Invalid API key. Check your GROQ_API_KEY.")
        elif status_code == 429:
            print("   → Rate limit exceeded. Wait and try again.")
        return False
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        return False


def check_text_generation() -> bool:
    """Test a minimal text generation call."""
    print_section("4. Text Generation Test")

    client = Groq(api_key=config.GROQ_API_KEY)
    prompt = "Reply with exactly one word: YES"

    print(f"   Sending: '{prompt}'")
    print(f"   Model: {config.GROQ_MODEL}")

    try:
        start = time.time()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.GROQ_MODEL,
            max_tokens=10,
            temperature=0.1,
        )
        elapsed = time.time() - start

        message = response.choices[0].message
        text = message.content.strip() if message.content else "(empty)"
        print(f"✅ PASS: Response received in {elapsed:.2f}s")
        print(f"   Response: '{text[:100]}'")
        return True

    except APIError as e:
        status_code = getattr(e, 'status_code', None)  # type: ignore[arg-type]
        print(f"❌ FAIL: API error: {e}")
        if status_code == 429:
            print("   → Rate limit exceeded.")
        return False
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        return False


def check_classification_prompt() -> bool:
    """Test a news classification prompt (mini test)."""
    print_section("5. Classification Prompt Test")

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import llm_interface

    prompt = (
        "You are a news classification expert. "
        "Classify the following news headline into one of these four categories: "
        "World, Sports, Business, Sci/Tech. "
        "Respond with ONLY the category name, nothing else.\n\n"
        "Headline: {title}\n\n"
        "Category:"
    )

    # Test with a clear sports headline
    title = "Lakers Defeat Celtics in Overtime Thriller"
    expected = "Sports"

    print(f"   Title: '{title}'")
    print(f"   Expected: {expected}")

    result = llm_interface.classify_text(prompt, title)

    if result == expected:
        print(f"✅ PASS: Correctly classified as '{result}'")
        return True
    elif result is None:
        print(f"❌ FAIL: Could not get a response (API error)")
        return False
    else:
        print(f"⚠️  UNEXPECTED: Got '{result}' instead of '{expected}'")
        return False


def main() -> None:
    """Run all health checks."""
    print("=" * 60)
    print("  EvoPrompt Optimizer — Groq API Health Check")
    print("=" * 60)

    results = {
        "API Key": check_api_key(),
        "SDK Import": check_sdk_import(),
        "Model Available": check_model_availability(),
        "Text Generation": check_text_generation(),
        "Classification": check_classification_prompt(),
    }

    print_section("Summary")
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("🎉 All checks passed! Groq API is ready for evolution.")
        print("   Run: python src/main.py")
    else:
        print("⚠️  Some checks failed. Fix the issues before running evolution.")
        print()
        print("   Common fixes:")
        print("   - Get API key from: https://console.groq.com/keys")
        print("   - Add to .env: GROQ_API_KEY=gsk_...")
        print("   - Check model name in .env")

    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
