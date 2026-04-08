"""
EvoPrompt Optimizer - API Health Check & Connection Test

Run this script to verify that:
1. The API key is configured
2. The google-genai SDK can authenticate
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

from google import genai
from google.genai import errors as genai_errors
import config


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check_api_key() -> bool:
    """Verify that an API key is configured."""
    print_section("1. API Key Check")

    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your_api_key_here":
        print("❌ FAIL: GEMINI_API_KEY is not set.")
        print("   → Edit .env and add your API key from:")
        print("     https://aistudio.google.com/app/apikey")
        return False

    masked = config.GEMINI_API_KEY[:4] + "..." + config.GEMINI_API_KEY[-4:]
    print(f"✅ PASS: API key is set ({masked})")
    return True


def check_sdk_import() -> bool:
    """Verify that the google-genai SDK can be imported."""
    print_section("2. SDK Import Check")

    try:
        from google import genai
        print(f"✅ PASS: google-genai SDK imported (version: {genai.__version__})")
        return True
    except ImportError as e:
        print(f"❌ FAIL: Cannot import google-genai SDK: {e}")
        return False


def check_model_availability() -> bool:
    """Verify that the configured model is available."""
    print_section("3. Model Availability Check")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    print(f"   Target model: {config.GEMINI_MODEL}")

    try:
        models = client.models.list()
        model_names = [m.name for m in models]

        # Check exact match
        full_target = f"models/{config.GEMINI_MODEL}"
        if full_target in model_names:
            print(f"✅ PASS: {config.GEMINI_MODEL} is available")
            return True

        # Check partial match (e.g., gemini-2.5-flash-latest alias)
        partial_matches = [m for m in model_names if config.GEMINI_MODEL in m]
        if partial_matches:
            print(f"⚠️  WARNING: Exact match not found, but found:")
            for m in partial_matches[:3]:
                print(f"      - {m}")
            print(f"   → The alias '{config.GEMINI_MODEL}' may still work at runtime.")
            return True

        print(f"❌ FAIL: {config.GEMINI_MODEL} not found in available models.")
        print(f"   Available Gemini models:")
        for m in [x for x in model_names if "gemini" in x.lower()][:5]:
            print(f"      - {m}")
        return False

    except genai_errors.APIError as e:
        print(f"❌ FAIL: API error listing models: {e}")
        if hasattr(e, 'code') and e.code == 401:
            print("   → Invalid API key. Check your GEMINI_API_KEY.")
        elif hasattr(e, 'code') and e.code == 429:
            print("   → Quota exceeded. Check your billing/usage limits.")
        return False
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        return False


def check_text_generation() -> bool:
    """Test a minimal text generation call."""
    print_section("4. Text Generation Test")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = "Reply with exactly one word: YES"

    print(f"   Sending: '{prompt}'")
    print(f"   Model: {config.GEMINI_MODEL}")

    try:
        start = time.time()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
        )
        elapsed = time.time() - start

        text = response.text.strip()
        print(f"✅ PASS: Response received in {elapsed:.2f}s")
        print(f"   Response: '{text[:100]}'")
        return True

    except genai_errors.APIError as e:
        elapsed = time.time() - start if 'start' in dir() else 0
        print(f"❌ FAIL: API error after {elapsed:.1f}s: {e}")
        if hasattr(e, 'code') and e.code == 429:
            print("   → Rate limit / quota exceeded.")
            print("   → Wait a few minutes and try again, or check billing.")
        elif hasattr(e, 'code') and e.code == 400:
            print("   → Bad request. Model name may be incorrect.")
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
    print("  EvoPrompt Optimizer — API Health Check")
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
        print("🎉 All checks passed! The API is ready for evolution.")
        print("   Run: python src/main.py")
    else:
        print("⚠️  Some checks failed. Fix the issues before running evolution.")
        print()
        print("   Common fixes:")
        print("   - Check API key in .env file")
        print("   - Get a new key from: https://aistudio.google.com/app/apikey")
        print("   - Check billing/quota at: https://console.cloud.google.com/")

    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
