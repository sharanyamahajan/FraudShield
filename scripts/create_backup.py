"""Create backup zip of models/, data/processed/, reports/, and data/raw/ outside the repository."""
import hashlib
import os
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    out_zip = Path("C:/dev/fraudshield_artifacts_backup.zip")
    logger_hashes = {}

    print(f"Creating backup archive at: {out_zip}")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in ["models", "data/processed", "reports", "data/raw"]:
            for dirpath, _, filenames in os.walk(folder):
                for filename in filenames:
                    p = Path(dirpath) / filename
                    rel = p.as_posix()
                    zf.write(p, rel)
                    h = hashlib.sha256(p.read_bytes()).hexdigest()
                    logger_hashes[rel] = h

    zip_hash = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    size_mb = out_zip.stat().st_size / (1024 * 1024)

    print(f"Backup zip successfully created at: {out_zip}")
    print(f"Zip file size: {size_mb:.2f} MB")
    print(f"Zip SHA-256: {zip_hash}")
    print("\nArchived File SHA-256 Hashes:")
    for k, v in sorted(logger_hashes.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
