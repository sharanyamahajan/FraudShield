import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

models_dir = Path("models")
reports_dir = Path("reports")
proc_dir = Path("data/processed")

# Load validation predictions of C1
val_probs_c1 = np.load(reports_dir / "m5_checkpoints" / "c1_s42_probs.npy")
df_val = pd.read_parquet(proc_dir / "val_set_b.parquet", columns=["isFraud"])
y_val = df_val["isFraud"].values

val_probs_clipped = np.clip(val_probs_c1, 1e-6, 1.0 - 1e-6)

# Fit Isotonic Calibrator
calibrator_c1 = IsotonicRegression(out_of_bounds="clip")
calibrator_c1.fit(val_probs_clipped, y_val)

# Save as models/calibrator_c1.pkl
cal_path = models_dir / "calibrator_c1.pkl"
with open(cal_path, "wb") as f:
    pickle.dump(calibrator_c1, f)

print(f"Successfully saved C1 calibrator to {cal_path}")
