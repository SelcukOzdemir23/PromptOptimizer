"""
EvoPrompt Optimizer - LLM Interface Module

Abstracts communication with the Google Gemini API. Handles prompt
submission, response parsing, error recovery with exponential backoff,
and rate limiting.
"""


def classify_text(prompt: str, text: str) -> str:
    """Send a prompt and text to the LLM, return predicted class label."""
    raise NotImplementedError("Phase 3: llm_interface not yet implemented.")


def classify_batch(prompt: str, texts: list[str]) -> list[str]:
    """Classify multiple texts with rate limiting and error handling."""
    raise NotImplementedError("Phase 3: llm_interface not yet implemented.")
