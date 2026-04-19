"""
EvoPrompt Optimizer - Data Management Module

Handles loading, preprocessing, and splitting of the AG News dataset.
Converts numeric labels to human-readable class names and prepares
data for the evolutionary pipeline.

The AG News dataset contains 120,000 training and 7,600 test samples
of news headlines across 4 categories: World, Sports, Business, Sci/Tech.

Only the Title (headline) column is used; Description is ignored.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import config


# ─── Class Label Mapping ─────────────────────────────────────────────────────

def get_class_labels() -> dict[int, str]:
    """
    Return the mapping from AG News numeric Class Index to human-readable names.

    Returns:
        dict: Mapping like {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    """
    return config.CLASS_LABELS


def get_label_mapping_reverse() -> dict[str, int]:
    """
    Return the reverse mapping: class name → numeric index.

    Used for converting model text responses back to numeric labels
    when needed internally.

    Returns:
        dict: Mapping like {'World': 1, 'Sports': 2, 'Business': 3, 'Sci/Tech': 4}
    """
    return {v: k for k, v in config.CLASS_LABELS.items()}


# ─── Dataset Loading ──────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """
    Load the AG News test dataset from CSV.

    Only the test split is used as specified in the requirements.
    Converts 'Class Index' to text labels and keeps only the 'Title' column.

    Returns:
        pd.DataFrame: DataFrame with columns ['title', 'label'] where
                      'title' is the news headline (str) and
                      'label' is the class name (str).
    """
    print("[DATA] Loading AG News test dataset...")

    df = pd.read_csv(config.TEST_CSV_PATH)

    # Map numeric labels to text
    df["label"] = df["Class Index"].map(config.CLASS_LABELS)

    # Keep only needed columns
    df = df[["Title", "label"]].copy()
    df.rename(columns={"Title": "title"}, inplace=True)

    # Drop any rows with missing values
    df.dropna(subset=["title", "label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[DATA] Loaded {len(df)} samples")
    print(f"[DATA] Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


# ─── Data Splitting ───────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    test_ratio: float | None = None,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into development pool and final test set.

    The development pool is used during the genetic algorithm evolution
    (fitness evaluation via mini-batch sampling). The final test set is
    held out and only used after evolution to measure generalization.

    Args:
        df: Full dataset DataFrame from load_dataset().
        test_ratio: Fraction for final test set (default: from config).
        random_seed: Random seed for reproducibility (default: from config).

    Returns:
        tuple: (dev_pool, test_set) as DataFrames with ['title', 'label'] columns.
    """
    if test_ratio is None:
        test_ratio = config.TEST_SPLIT_RATIO
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    print(f"[DATA] Splitting dataset: {test_ratio:.0%} for final test...")

    dev_pool, test_set = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=df["label"],  # Preserve class distribution
    )

    dev_pool = dev_pool.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    print(f"[DATA] Development pool: {len(dev_pool)} samples")
    print(f"[DATA] Final test set:   {len(test_set)} samples")

    return dev_pool, test_set


# ─── Data Persistence ─────────────────────────────────────────────────────────

def save_processed_data(dev_pool: pd.DataFrame, test_set: pd.DataFrame) -> None:
    """
    Save the split datasets to the processed data directory.

    This allows reloading without re-splitting and provides reproducibility.

    Args:
        dev_pool: Development pool DataFrame.
        test_set: Final test set DataFrame.
    """
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dev_path = config.PROCESSED_DATA_DIR / "dev_pool.csv"
    test_path = config.PROCESSED_DATA_DIR / "test_set.csv"

    dev_pool.to_csv(dev_path, index=False)
    test_set.to_csv(test_path, index=False)

    print(f"[DATA] Saved dev_pool to {dev_path}")
    print(f"[DATA] Saved test_set  to {test_path}")


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved processed datasets.

    Returns:
        tuple: (dev_pool, test_set) as DataFrames.

    Raises:
        FileNotFoundError: If processed data files do not exist.
    """
    dev_path = config.PROCESSED_DATA_DIR / "dev_pool.csv"
    test_path = config.PROCESSED_DATA_DIR / "test_set.csv"

    dev_pool = pd.read_csv(dev_path)
    test_set = pd.read_csv(test_path)

    print(f"[DATA] Loaded processed data: {len(dev_pool)} dev, {len(test_set)} test")
    return dev_pool, test_set


# ─── Convenience: Full Pipeline ───────────────────────────────────────────────

def prepare_dataset(force_reprocess: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load raw data, split, save, and return datasets.

    If processed data already exists (and force_reprocess is False),
    loads from disk instead of re-processing.

    Args:
        force_reprocess: If True, re-split even if processed data exists.

    Returns:
        tuple: (dev_pool, test_set) ready for use in evolution.
    """
    dev_path = config.PROCESSED_DATA_DIR / "dev_pool.csv"
    test_path = config.PROCESSED_DATA_DIR / "test_set.csv"

    if not force_reprocess and dev_path.exists() and test_path.exists():
        print("[DATA] Found existing processed data, loading from disk...")
        return load_processed_data()

    df = load_dataset()
    dev_pool, test_set = split_data(df)
    save_processed_data(dev_pool, test_set)

    return dev_pool, test_set
