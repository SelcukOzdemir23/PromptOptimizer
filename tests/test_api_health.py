"""
EvoPrompt Optimizer - API Health Check & Connection Test

Run this script to verify that:
1. Ollama is running and accessible
2. The target model is installed and accessible
3. A basic text generation call succeeds

Usage:
    source venv/bin/activate
    python tests/test_api_health.py
"""

import sys
import time
from pathlib import Path
import httpx

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config

OLLAMA_URL = "http://localhost:11434"


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check_ollama_running() -> bool:
    """Verify that Ollama is running."""
    print_section("1. Ollama Server Check")

    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        response.raise_for_status()
        print("✅ PASS: Ollama is running and accessible")
        return True
    except httpx.ConnectError:
        print("❌ FAIL: Cannot connect to Ollama at localhost:11434")
        print("   → Start Ollama: ollama serve")
        return False
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        return False


def check_model_available() -> bool:
    """Verify that the configured model is available."""
    print_section("2. Model Availability Check")

    print(f"   Target model: {config.OLLAMA_MODEL}")

    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        if config.OLLAMA_MODEL in model_names:
            print(f"✅ PASS: {config.OLLAMA_MODEL} is installed")
            return True
        else:
            print(f"❌ FAIL: {config.OLLAMA_MODEL} not found")
            print(f"   Available models: {model_names}")
            print(f"   → Install: ollama pull {config.OLLAMA_MODEL}")
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def check_text_generation() -> bool:
    """Test a minimal text generation call."""
    print_section("3. Text Generation Test")

    prompt = "Reply with exactly one word: YES"
    print(f"   Sending: '{prompt}'")
    print(f"   Model: {config.OLLAMA_MODEL}")

    try:
        start = time.time()
        response = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 10},
            },
            timeout=120.0,  # First call may need model load
        )
        elapsed = time.time() - start

        text = response.json().get("message", {}).get("content", "").strip()
        print(f"✅ PASS: Response received in {elapsed:.2f}s")
        print(f"   Response: '{text[:100]}'")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def check_classification_prompt() -> bool:
    """Test a news classification prompt."""
    print_section("4. Classification Prompt Test")

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

    title = "Lakers Defeat Celtics in Overtime Thriller"
    expected = "Sports"

    print(f"   Title: '{title}'")
    print(f"   Expected: {expected}")

    result = llm_interface.classify_text(prompt, title)

    if result == expected:
        print(f"✅ PASS: Correctly classified as '{result}'")
        return True
    elif result is None:
        print(f"❌ FAIL: Could not get a response")
        return False
    else:
        print(f"⚠️  UNEXPECTED: Got '{result}' instead of '{expected}'")
        return False


def main() -> None:
    """Run all health checks."""
    print("=" * 60)
    print("  EvoPrompt Optimizer — Ollama Health Check")
    print("=" * 60)

    results = {
        "Ollama Running": check_ollama_running(),
        "Model Available": check_model_available(),
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
        print("🎉 All checks passed! Ollama is ready for evolution.")
        print("   Run: python src/main.py")
    else:
        print("⚠️  Some checks failed.")
        print()
        print("   Common fixes:")
        print("   - Start Ollama: ollama serve")
        print("   - Install model: ollama pull llama3.1:8b")

    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
