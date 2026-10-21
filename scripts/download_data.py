"""Download and extract IEEE-CIS Fraud Detection dataset files for FraudShield."""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fraudshield.download")

COMPETITION = "ieee-fraud-detection"
TARGET_FILES = ["train_transaction.csv", "train_identity.csv"]


def download_and_extract(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    all_present = all((raw_dir / f).exists() for f in TARGET_FILES)
    if all_present:
        logger.info("All target files already exist in %s. Skipping download.", raw_dir)
        for f in TARGET_FILES:
            size_mb = (raw_dir / f).stat().st_size / (1024 * 1024)
            logger.info(" - %s (%.2f MB)", f, size_mb)
        return

    logger.info("Downloading dataset files from Kaggle competition '%s'...", COMPETITION)
    for target in TARGET_FILES:
        target_path = raw_dir / target
        if target_path.exists():
            logger.info("File %s already exists. Skipping download.", target)
            continue

        zip_name = f"{target}.zip"
        zip_path = raw_dir / zip_name

        logger.info("Downloading %s ...", target)
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            target,
            "-p",
            str(raw_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("Kaggle download failed: %s\n%s", res.stdout, res.stderr)
            raise RuntimeError(f"Failed to download {target} from Kaggle.")

        # Check if downloaded as zip
        if zip_path.exists():
            logger.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(raw_dir)
            zip_path.unlink()  # remove zip to conserve disk space
        elif not target_path.exists():
            # Check if any zip was created
            zips = list(raw_dir.glob("*.zip"))
            for z in zips:
                logger.info("Extracting %s ...", z.name)
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(raw_dir)
                z.unlink()

        if not target_path.exists():
            raise FileNotFoundError(f"Expected file {target_path} not found after extraction.")

        size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info("Successfully acquired %s (%.2f MB)", target, size_mb)


def main():
    parser = argparse.ArgumentParser(description="Download IEEE-CIS Fraud Detection dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Target directory for raw data",
    )
    args = parser.parse_args()
    download_and_extract(args.output_dir)


if __name__ == "__main__":
    main()
