"""
EvoPrompt Optimizer - LLM Interface Module

Abstracts communication with Ollama (local LLM inference).
Ollama runs locally at http://localhost:11434 — no API key needed, no rate limits.

Handles:
- Prompt submission with headline classification
- Response parsing and label normalization
- Retry on transient errors (connection failures)
- Batch classification with progress reporting
"""

import time
import logging
import httpx

import config

# Configure module logger
logger = logging.getLogger(__name__)

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/chat"


# ─── Response Parsing ─────────────────────────────────────────────────────────

def _parse_response(response_text: str) -> str | None:
    """
    Parse the LLM's raw text response into a valid class label.

    Strips whitespace, removes common prefixes/suffixes, and matches
    against known class labels (case-insensitive).

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        str | None: Matched class label (e.g., 'World') or None if unparseable.
    """
    cleaned = response_text.strip().strip("\"'").strip()

    # Try exact match (case-insensitive)
    label_map_lower = {v.lower(): v for v in config.CLASS_LABELS.values()}
    if cleaned.lower() in label_map_lower:
        return label_map_lower[cleaned.lower()]

    # Try partial match (e.g., "The category is World" → "World")
    for lower_name, full_name in label_map_lower.items():
        if lower_name in cleaned.lower():
            return full_name

    logger.warning(f"Could not parse response: '{response_text}'")
    return None


# ─── Single Classification ────────────────────────────────────────────────────

def classify_text(prompt: str, title: str) -> str | None:
    """
    Classify a single news headline using the local Ollama model.

    Fills the {title} placeholder in the prompt, sends it to Ollama,
    and parses the response into a class label.

    Implements retry on connection errors.

    Args:
        prompt: Prompt template with {title} placeholder.
        title: News headline to classify.

    Returns:
        str | None: Predicted class label or None if all retries fail.
    """
    full_prompt = prompt.format(title=title)

    last_error = None
    for attempt in range(1, config.API_MAX_RETRIES + 1):
        try:
            if attempt > 1:
                backoff = config.API_BACKOFF_FACTOR ** (attempt - 1)
                wait_time = max(config.API_CALL_DELAY, backoff)
                logger.info(f"Retry {attempt}/{config.API_MAX_RETRIES}, "
                            f"waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                time.sleep(config.API_CALL_DELAY)

            # Send request to Ollama REST API
            response = httpx.post(
                OLLAMA_URL,
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "user", "content": full_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10,
                    },
                },
                timeout=60.0,  # First call may need model load time
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("message", {}).get("content", "")

            if not content:
                logger.warning(f"Empty response from Ollama for title: '{title}'")
                return None

            return _parse_response(content)

        except httpx.ConnectError as e:
            last_error = e
            logger.error(f"Connection error on attempt {attempt}: {e}")
            if attempt < config.API_MAX_RETRIES:
                time.sleep(config.API_CALL_DELAY * 2)
            continue

        except httpx.HTTPStatusError as e:
            last_error = e
            logger.error(f"HTTP error on attempt {attempt}: {e}")
            break

        except Exception as e:
            last_error = e
            logger.error(f"Unexpected error on attempt {attempt}: {e}")
            break

    logger.error(f"All {config.API_MAX_RETRIES} retries failed for title: '{title}'")
    return None


# ─── Batch Classification ─────────────────────────────────────────────────────

def classify_batch(prompt: str, titles: list[str]) -> list[str | None]:
    """
    Classify multiple headlines with rate limiting and progress reporting.

    Args:
        prompt: Prompt template with {title} placeholder.
        titles: List of news headlines to classify.

    Returns:
        list[str | None]: Predicted class labels (None for failed calls).
    """
    results = []
    total = len(titles)

    for i, title in enumerate(titles, 1):
        label = classify_text(prompt, title)
        results.append(label)

        # Progress logging every 10 items
        if i % 10 == 0 or i == total:
            success = sum(1 for r in results if r is not None)
            logger.info(f"[LLM] Progress: {i}/{total} ({i/total:.0%}), "
                        f"success: {success}/{i}")

    return results


# ─── Accuracy Evaluation ──────────────────────────────────────────────────────

def evaluate_prompt(
    prompt: str,
    titles: list[str],
    labels: list[str],
) -> float:
    """
    Evaluate a prompt's accuracy on a labeled dataset.

    Classifies each title and compares predictions to ground truth.

    Args:
        prompt: Prompt template with {title} placeholder.
        titles: List of news headlines.
        labels: List of ground truth class labels.

    Returns:
        float: Accuracy score (correct predictions / total predictions).
               Failed API calls count as incorrect.
    """
    if len(titles) != len(labels):
        raise ValueError("titles and labels must have the same length")

    predictions = classify_batch(prompt, titles)

    correct = sum(
        1 for pred, true in zip(predictions, labels)
        if pred is not None and pred == true
    )

    accuracy = correct / len(labels) if labels else 0.0
    logger.info(f"[LLM] Accuracy: {accuracy:.4f} ({correct}/{len(labels)})")
    return accuracy
