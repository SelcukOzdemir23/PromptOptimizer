"""
EvoPrompt Optimizer - Central Configuration Module

Loads all configuration values from environment variables and provides
them as typed constants for the rest of the application.

Project: HZ-2026-001 | Computational Intelligence Midterm
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"

# ─── Dataset ─────────────────────────────────────────────────────────────────

# AG News dataset class mapping (Class Index → Label name)
CLASS_LABELS = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

# Source dataset paths (relative to project root)
TRAIN_CSV_PATH = BASE_DIR / "dataset" / "train.csv"
TEST_CSV_PATH = BASE_DIR / "dataset" / "test.csv"

# Data split ratio for development pool vs final test set
TEST_SPLIT_RATIO = float(os.getenv("TEST_SPLIT_RATIO", "0.5"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ─── LLM / Gemini Configuration ──────────────────────────────────────────────

# NOTE: Uses the new Google GenAI SDK (google-genai >= 1.0.0)
# The old google-generativeai SDK is deprecated since late 2024.
# API key is read automatically from GEMINI_API_KEY environment variable.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Rate limiting & retry
API_CALL_DELAY = float(os.getenv("API_CALL_DELAY", "4.0"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
API_BACKOFF_FACTOR = float(os.getenv("API_BACKOFF_FACTOR", "2.0"))

# ─── Evolution Parameters ────────────────────────────────────────────────────

POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", "20"))
GENERATIONS = int(os.getenv("GENERATIONS", "10"))
MUTATION_PROBABILITY = float(os.getenv("MUTATION_PROBABILITY", "0.2"))
CROSSOVER_PROBABILITY = float(os.getenv("CROSSOVER_PROBABILITY", "0.8"))
TOURNAMENT_SIZE = int(os.getenv("TOURNAMENT_SIZE", "3"))
MINI_BATCH_SIZE = int(os.getenv("MINI_BATCH_SIZE", "50"))

# ─── Human-written Initial Prompt ────────────────────────────────────────────
# This is the seed prompt that the genetic algorithm will evolve.
# Format: A template with {title} placeholder for the news headline.

INITIAL_PROMPT = (
    "You are a news classification expert. "
    "Classify the following news headline into one of these four categories: "
    "World, Sports, Business, Sci/Tech. "
    "Respond with ONLY the category name, nothing else.\n\n"
    "Headline: {title}\n\n"
    "Category:"
)

# ─── Validation ──────────────────────────────────────────────────────────────

def validate_config() -> None:
    """
    Validate that all required configuration values are present and valid.
    Raises ValueError if any critical configuration is missing or invalid.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Please copy .env.example to .env and add your API key."
        )

    if POPULATION_SIZE < 2:
        raise ValueError("POPULATION_SIZE must be at least 2.")

    if GENERATIONS < 1:
        raise ValueError("GENERATIONS must be at least 1.")

    if not (0.0 <= MUTATION_PROBABILITY <= 1.0):
        raise ValueError("MUTATION_PROBABILITY must be between 0.0 and 1.0.")

    if not (0.0 <= CROSSOVER_PROBABILITY <= 1.0):
        raise ValueError("CROSSOVER_PROBABILITY must be between 0.0 and 1.0.")

    if MINI_BATCH_SIZE < 1:
        raise ValueError("MINI_BATCH_SIZE must be at least 1.")

    # Ensure output directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
