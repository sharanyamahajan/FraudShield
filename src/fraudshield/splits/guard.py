"""Split loading guard to prevent premature test-set leakage or access.

Test set isolation architecture:
- The held-out test split is sealed throughout all training and validation phases (Milestones 0 to 4).
- Access to the test split cannot be unlocked via arbitrary boolean flags.
- Access requires the internal FINAL_EVALUATION_AUTH_TOKEN, which is scoped strictly
  to the final Milestone 5 evaluation module (src/fraudshield/evaluation/final_evaluator.py).
- Any attempt to load the test split without this token raises PrematureTestAccessError.
"""

import logging
import os
import secrets
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger("fraudshield.splits.guard")

# Ephemeral secret token generated for the process lifecycle, accessible only to final evaluation
_EVALUATION_TOKEN = os.environ.get("FRAUDSHIELD_FINAL_EVAL_TOKEN", secrets.token_hex(32))


class PrematureTestAccessError(PermissionError):
    """Raised when an operation attempts to access the held-out test split prematurely."""
    pass


def get_final_eval_token_for_authorized_runner(authorized_caller_file: str) -> str:
    """Return the authorization token ONLY if the caller is the verified final evaluation module."""
    caller_path = Path(authorized_caller_file).resolve()
    # Must be within final evaluation script
    if "evaluate_test" not in caller_path.name and "final_evaluator" not in caller_path.name:
        raise PermissionError(f"Caller '{caller_path.name}' is not authorized to request test evaluation token.")
    return _EVALUATION_TOKEN


def load_split_parquet(
    split_name: str,
    feature_set: str = "a",
    base_dir: Path = Path("data/processed"),
    auth_token: Optional[str] = None,
) -> pd.DataFrame:
    """Load processed parquet for a given split, strictly enforcing test set isolation.
    
    Parameters
    ----------
    split_name : str
        One of 'train', 'val', or 'test'.
    feature_set : str
        'a', 'b', 'c1', or 'c2'.
    base_dir : Path
        Directory containing processed parquets.
    auth_token : str, optional
        Must match the internal _EVALUATION_TOKEN. Only available to the final evaluation runner.
    """
    split_lower = split_name.lower().strip()
    if split_lower == "test":
        if auth_token is None or auth_token != _EVALUATION_TOKEN:
            msg = (
                "SECURITY VIOLATION: Premature access to the held-out test split is strictly forbidden. "
                "Normal training and baseline scripts cannot unlock the test split. "
                "The test split may only be accessed once during final Milestone 5 reporting "
                "via the authorized final evaluation module."
            )
            logger.error(msg)
            raise PrematureTestAccessError(msg)

    target_file = base_dir / f"{split_lower}_set_{feature_set.lower()}.parquet"
    if not target_file.exists():
        raise FileNotFoundError(f"Processed file not found: {target_file}")

    logger.info("Loading split '%s' for Set %s from %s ...", split_name, feature_set.upper(), target_file)
    return pd.read_parquet(target_file)
