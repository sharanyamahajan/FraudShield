#!/usr/bin/env bash
set -euo pipefail

# Download script for IEEE-CIS Fraud Detection dataset
# Usage: ./scripts/download_data.sh

RAW_DIR="data/raw"
mkdir -p "${RAW_DIR}"

echo "Checking Kaggle API credentials..."
if ! command -v kaggle &> /dev/null; then
    echo "Using python -m kaggle..."
    KAGGLE_CMD="python -m kaggle"
else
    KAGGLE_CMD="kaggle"
fi

echo "Downloading train_transaction.csv and train_identity.csv into ${RAW_DIR}..."
${KAGGLE_CMD} competitions download -c ieee-fraud-detection -f train_transaction.csv -p "${RAW_DIR}"
${KAGGLE_CMD} competitions download -c ieee-fraud-detection -f train_identity.csv -p "${RAW_DIR}"

echo "Extracting archives..."
for f in "${RAW_DIR}"/*.zip; do
    if [ -f "$f" ]; then
        echo "Unzipping $f..."
        unzip -o "$f" -d "${RAW_DIR}"
        rm -f "$f"
    fi
done

echo "Download and extraction complete. Raw files present in ${RAW_DIR}:"
ls -lh "${RAW_DIR}"
