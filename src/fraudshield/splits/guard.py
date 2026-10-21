"""Split loading guard to prevent premature test-set leakage or access."""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger("fraudshield.splits.guard")


class PrematureTestAccessError(PermissionError):
    """Raised when an operation attempts to access the held-out test split prematurely."""
    pass


def load_split_parquet(
    split_name: str,
    feature_set: str = "a",
    base_dir: Path = Path("data/processed"),
    allow_test_access: bool = False,
) -> pd.DataFrame:
    """Load processed parquet for a given split, enforcing test set isolation.
    
    Parameters
    ----------
    split_name : str
        One of 'train', 'val', or 'test'.
    feature_set : str
        'a', 'b', or 'c'.
    base_dir : Path
        Directory containing processed parquets.
    allow_test_access : bool
        Must be explicitly True (only allowed in final Milestone 5 evaluation).
    """
    split_lower = split_name.lower().strip()
    if split_lower == "test" and not allow_test_access:
        msg = (
            "SECURITY VIOLATION: Premature access to the held-out test split is strictly forbidden. "
            "The test split may only be accessed once during final Milestone 5 reporting."
        )
        logger.error(msg)
        raise PrematureTestAccessError(msg)

    target_file = base_dir / f"{split_lower}_set_{feature_set.lower()}.parquet"
    if not target_file.exists():
        raise FileNotFoundError(f"Processed file not found: {target_file}")

    logger.info("Loading split '%s' for Set %s from %s ...", split_name, feature_set.upper(), target_file)
    return pd.read_parquet(target_file)
