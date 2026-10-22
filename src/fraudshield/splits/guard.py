"""Split loading guard to prevent premature test-set leakage or access.

Test set isolation architecture:
- The held-out test split is sealed throughout all training and validation phases (Milestones 0 to 4).
- Access to the test split cannot be unlocked via caller-supplied flags, arguments, or filenames.
- The guard inspects the real Python call stack (using inspect.stack()).
- Access is permitted ONLY if an authorized final evaluation module
  (e.g., 'final_evaluator.py' or 'evaluate_test.py') is present in the call stack.
- Every authorized access is permanently logged to reports/test_access_log.txt.
- NOTE: As documented in docs/METHODOLOGY.md, this mechanism is an accident guard designed
  to prevent inadvertent evaluation leakage, not an operating system security boundary.
"""

import datetime
import inspect
import logging
from pathlib import Path
from typing import Tuple, Set
import pandas as pd

logger = logging.getLogger("fraudshield.splits.guard")

# Only these modules in the call stack are authorized to load the held-out test split
AUTHORIZED_CALLER_MODULES: Set[str] = {
    "final_evaluator.py",
    "evaluate_test.py",
}


class PrematureTestAccessError(PermissionError):
    """Raised when an operation attempts to access the held-out test split prematurely."""
    pass


def _inspect_call_stack_for_authorization() -> Tuple[bool, str]:
    """Inspect the live Python call stack to verify if an authorized evaluator is calling.
    
    Returns
    -------
    Tuple[bool, str]
        (is_authorized, caller_description)
    """
    stack = inspect.stack()
    # Check frames starting from caller of load_split_parquet
    for frame_info in stack[1:]:
        frame_filename = Path(frame_info.filename).name
        if frame_filename in AUTHORIZED_CALLER_MODULES:
            return True, str(Path(frame_info.filename).resolve())

    # Get immediate caller path for diagnostic message
    caller_path = str(Path(stack[1].filename).resolve()) if len(stack) > 1 else "unknown"
    return False, caller_path


def _log_authorized_test_access(
    caller_path: str,
    split_name: str,
    feature_set: str,
    file_path: Path,
    log_file: Optional[Path] = None,
) -> None:
    """Log authorized test split access to reports/test_access_log.txt or a specified log file."""
    if log_file is None:
        log_file = Path("reports/test_access_log.txt")
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry = (
        f"[{timestamp}] AUTHORIZED TEST ACCESS: "
        f"split={split_name}, feature_set={feature_set}, target={file_path.as_posix()}, caller={caller_path}\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry)
    logger.info("Logged authorized test split access from %s to %s", caller_path, log_file)


def load_split_parquet(
    split_name: str,
    feature_set: str = "a",
    base_dir: Path = Path("data/processed"),
    log_file: Optional[Path] = None,
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
    log_file : Optional[Path]
        Log file path for authorized test split access (defaults to reports/test_access_log.txt).
    """
    split_lower = split_name.lower().strip()
    target_file = Path(base_dir) / f"{split_lower}_set_{feature_set.lower()}.parquet"

    if split_lower == "test":
        is_authorized, caller_path = _inspect_call_stack_for_authorization()
        if not is_authorized:
            msg = (
                f"SECURITY VIOLATION: Premature access to the held-out test split is strictly forbidden. "
                f"Caller '{caller_path}' is not an authorized final evaluation module. "
                f"Access is restricted to {AUTHORIZED_CALLER_MODULES}."
            )
            logger.error(msg)
            raise PrematureTestAccessError(msg)

        _log_authorized_test_access(caller_path, split_name, feature_set, target_file, log_file=log_file)

    if not target_file.exists():
        raise FileNotFoundError(f"Processed file not found: {target_file}")

    logger.info("Loading split '%s' for Set %s from %s ...", split_name, feature_set.upper(), target_file)
    return pd.read_parquet(target_file)
