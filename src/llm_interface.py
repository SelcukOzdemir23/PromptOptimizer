"""
EvoPrompt Optimizer - LLM Interface Module

Abstracts communication with the Groq API (groq SDK).
Groq provides extremely fast inference with a very generous free tier.

Handles:
- Prompt submission with headline classification
- Response parsing and label normalization
- Exponential backoff retry on transient errors
- Rate limiting (configurable delay between calls)
- Batch classification with error handling
"""

import time
import logging
from groq import Groq
from groq import APIError

import config

# Configure module logger
logger = logging.getLogger(__name__)


# ─── Client Initialization ────────────────────────────────────────────────────

def _get_client() -> Groq:
    """
    Create and return a Groq API client.

    max_retries=0: We handle retries manually in classify_text()
    to avoid double-retry storms with Groq's built-in retry.

    The client reads GROQ_API_KEY from environment variables
    (automatically loaded by python-dotenv from .env file).

    Returns:
        Groq: Configured Groq API client instance.
    """
    return Groq(api_key=config.GROQ_API_KEY, max_retries=0)


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
    Classify a single news headline using the LLM.

    Fills the {title} placeholder in the prompt, sends it to the model,
    and parses the response into a class label.

    Implements exponential backoff retry on transient API errors.

    Args:
        prompt: Prompt template with {title} placeholder.
        title: News headline to classify.

    Returns:
        str | None: Predicted class label or None if all retries fail.
    """
    client = _get_client()
    full_prompt = prompt.format(title=title)

    last_error = None
    for attempt in range(1, config.API_MAX_RETRIES + 1):
        try:
            # Rate limiting: wait between API calls
            if attempt > 1:
                backoff = config.API_BACKOFF_FACTOR ** (attempt - 1)
                wait_time = max(config.API_CALL_DELAY, backoff)
                logger.info(f"Retry {attempt}/{config.API_MAX_RETRIES}, "
                            f"waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                time.sleep(config.API_CALL_DELAY)

            # Send request using Groq SDK
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                model=config.GROQ_MODEL,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=10,    # We only need one word (the category)
            )

            response_text = chat_completion.choices[0].message.content
            if response_text is None:
                logger.warning("Got empty response from LLM")
                return None
            return _parse_response(response_text)

        except APIError as e:
            last_error = e
            logger.error(f"API error on attempt {attempt}: {e}")
            status_code = getattr(e, 'status_code', None)

            # Rate limit errors (429) — wait longer and retry
            if status_code == 429:
                continue
            # Server errors — retry
            if status_code and 500 <= status_code < 600:
                continue
            # Other errors — fail immediately
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
