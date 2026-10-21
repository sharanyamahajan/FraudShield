"""Skeleton tests verifying repository setup, configuration validity, and CLI structure."""

from pathlib import Path
import pytest
import yaml
import fraudshield
from fraudshield.cli import main


def test_package_metadata():
    """Verify package version and basic import."""
    assert fraudshield.__version__ == "0.1.0"


def test_config_file_exists_and_parses():
    """Verify configs/config.yaml is present and conforms to expected structure."""
    config_path = Path("configs/config.yaml")
    assert config_path.exists(), "configs/config.yaml is missing"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    assert isinstance(config, dict)
    assert config["seed"] == 42
    assert "paths" in config
    assert "splits" in config
    assert "features" in config
    assert "models" in config
    assert "evaluation" in config
    
    # Assert split ratios sum to 1.0
    splits = config["splits"]
    total_ratio = splits["train_ratio"] + splits["val_ratio"] + splits["test_ratio"]
    assert pytest.approx(total_ratio, 1e-5) == 1.0
    
    # Assert review budgets are ascending and within (0, 1)
    budgets = config["evaluation"]["review_budgets"]
    assert len(budgets) > 0
    assert all(0 < b < 1 for b in budgets)
    assert sorted(budgets) == budgets


def test_expected_directories_exist():
    """Verify that required workspace directories exist."""
    expected_dirs = [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "src/fraudshield",
        "src/fraudshield/data",
        "src/fraudshield/features",
        "src/fraudshield/splits",
        "src/fraudshield/models",
        "src/fraudshield/evaluation",
        "src/fraudshield/explain",
        "src/fraudshield/api",
        "scripts",
        "tests",
        "notebooks",
        "docs",
        "reports",
        "app",
    ]
    for d in expected_dirs:
        p = Path(d)
        assert p.exists() and p.is_dir(), f"Expected directory '{d}' does not exist."


def test_cli_subcommands():
    """Verify CLI argument parser handles each command without unhandled exception."""
    commands = ["data", "features", "train", "evaluate", "test"]
    for cmd in commands:
        # CLI returns 0 for help and successful dispatch
        with pytest.raises(SystemExit) as exc_info:
            main([cmd, "--help"])
        assert exc_info.value.code == 0
