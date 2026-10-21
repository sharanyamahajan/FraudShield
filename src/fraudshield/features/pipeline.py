"""Feature extraction and preprocessing pipelines for Feature Sets A and B.

Explicit schema partitioning:
- Set A: 12 numeric features + 5 categorical features = 17 features.
- Set B: 403 numeric features + 31 categorical features = 434 features.

All encoders, imputers, and scalers are fitted STRICTLY on the TRAIN split.
- TreePreprocessor:
  - Categoricals: train-fitted ordinal integers (unseen -> -1.0, missing -> -2.0).
  - Numerics: preserved as float32 with NaNs intact for native tree split routing.
- LinearPreprocessor:
  - Categoricals: train-fitted relative frequency encoding (freq = count / N_train),
    scaled with train-fitted mean/std, plus explicit _is_unseen and _isna indicators (3 cols/cat).
  - Numerics: median imputation fitted on train, standard scaled on train, plus explicit _isna indicator (2 cols/num).
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("fraudshield.features.pipeline")

# Columns that must NEVER be fed into ML models as feature inputs
METADATA_AND_TARGET_COLS = ["TransactionID", "TransactionDT", "isFraud"]

# Canonical categorical columns across IEEE-CIS
ALL_RAW_CATEGORICAL_COLS = [
    # Set A categoricals (5)
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    # Set B additional categoricals (26)
    "DeviceType",
    "DeviceInfo",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
]

SET_A_RAW_NUMERIC = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
]

SET_A_RAW_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
]

SET_A_ENGINEERED_NUMERIC = [
    "log_amount",
    "amount_cents",
    "hour_of_day_proxy",
]


def get_model_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return model feature column names, guaranteeing exclusion of TransactionID, TransactionDT, and isFraud."""
    return [col for col in df.columns if col not in METADATA_AND_TARGET_COLS]


def extract_set_a_raw(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    """Extract and engineer Set A features from raw dataframe.
    
    Guarantees that NO V, C, D, M, or identity columns are included.
    Returns 17 un-encoded feature columns (12 numeric, 5 categorical).
    """
    df_out = pd.DataFrame(index=df.index)

    # 1. Base numeric columns
    for col in SET_A_RAW_NUMERIC:
        if col in df.columns:
            df_out[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        else:
            df_out[col] = np.nan

    # 2. Base categorical columns (preserved as string/object with uniform NaN)
    for col in SET_A_RAW_CATEGORICAL:
        if col in df.columns:
            s = df[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
            df_out[col] = s
        else:
            df_out[col] = np.nan

    # 3. Simple non-leaking transforms
    amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0.0).values
    df_out["log_amount"] = np.log1p(np.maximum(0.0, amt)).astype(np.float32)
    df_out["amount_cents"] = np.mod(amt, 1.0).astype(np.float32)

    # NOTE: TransactionDT is a relative offset in seconds from an unrevealed reference epoch,
    # so hour_of_day_proxy is only an approximation of diurnal cycle and must be documented as such.
    dt = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).values
    df_out["hour_of_day_proxy"] = ((dt // 3600) % 24).astype(np.float32)

    # Verification: Ensure no V, C, D, M, id_*, or Device columns leak into Set A
    forbidden = [
        c for c in df_out.columns
        if any(c.startswith(prefix) for prefix in ["V", "C", "D", "M", "id_"])
        or c in ["DeviceType", "DeviceInfo"]
        or c in METADATA_AND_TARGET_COLS
    ]
    if forbidden:
        raise ValueError(f"Set A contains forbidden columns: {forbidden}")

    return df_out


def extract_set_b_raw(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    """Extract Set B features: Set A + provided C, D, M, V and identity columns.
    
    Returns 434 un-encoded feature columns (403 numeric, 31 categorical).
    """
    df_a = extract_set_a_raw(df, time_col=time_col)
    df_b = df_a.copy()

    # Identify all Set B eligible columns present in df
    set_b_cols = [
        col for col in df.columns
        if col not in df_a.columns
        and col not in METADATA_AND_TARGET_COLS
        and (
            any(col.startswith(p) for p in ["C", "D", "V"])
            or (col.startswith("M") and len(col) <= 3 and col[1:].isdigit())
            or col.startswith("id_")
            or col in ["DeviceType", "DeviceInfo"]
        )
    ]

    for col in set_b_cols:
        if col in ALL_RAW_CATEGORICAL_COLS or not pd.api.types.is_numeric_dtype(df[col]):
            df_b[col] = df[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
        else:
            df_b[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    return df_b


extract_set_b = extract_set_b_raw


class TreePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for tree-based models (XGBoost).
    
    Fits categorical ordinal/integer encoding STRICTLY on the train split.
    Unseen categories in validation/test are mapped to -1.
    Missing categorical values are mapped to -2.
    Numeric values and NaNs are preserved for native handling by XGBoost.
    """

    def __init__(self):
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.feature_names_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.numeric_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoding strictly on training data."""
        clean_cols = [c for c in X.columns if c not in METADATA_AND_TARGET_COLS]
        self.feature_names_ = clean_cols
        self.categorical_cols_ = [
            c for c in clean_cols if c in ALL_RAW_CATEGORICAL_COLS or not pd.api.types.is_numeric_dtype(X[c])
        ]
        self.numeric_cols_ = [
            c for c in clean_cols if c not in self.categorical_cols_
        ]

        self.category_maps_ = {}
        for col in self.categorical_cols_:
            valid_cats = X[col].dropna().unique()
            cat_map = {cat: i for i, cat in enumerate(sorted(valid_cats))}
            self.category_maps_[col] = cat_map

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the train-fitted category dictionaries."""
        data_dict = {}

        # Pass through numerics
        for col in self.numeric_cols_:
            if col in X.columns:
                data_dict[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32).values
            else:
                data_dict[col] = np.full(len(X), np.nan, dtype=np.float32)

        # Encode categoricals: known -> integer, unseen -> -1, missing -> -2
        for col in self.categorical_cols_:
            if col in X.columns:
                cat_map = self.category_maps_[col]
                mapped = X[col].map(cat_map).fillna(-1)
                mapped = np.where(X[col].isna(), -2, mapped)
                data_dict[col] = mapped.astype(np.float32)
            else:
                data_dict[col] = np.full(len(X), -2.0, dtype=np.float32)

        return pd.DataFrame(data_dict, index=X.index)


class LinearPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for linear models (Logistic Regression).
    
    Explicit Partitioning:
    - Categorical features (identified by ALL_RAW_CATEGORICAL_COLS):
      - Relative frequency encoding fitted on TRAIN: freq(c) = count(c) / N_train.
      - Frequency feature is standard-scaled using train mean and std: (freq - mean) / std.
      - Explicit _is_unseen indicator (1.0 if category was never seen in train, 0.0 otherwise).
      - Explicit _isna indicator (1.0 if category was missing, 0.0 otherwise).
      - Total columns per categorical = 3 columns.
    
    - Numeric features:
      - Median imputation fitted on TRAIN.
      - Standard scaling fitted on TRAIN: (val - mean) / std.
      - Explicit _isna indicator (1.0 if value was missing, 0.0 otherwise).
      - Total columns per numeric = 2 columns.
    """

    def __init__(self):
        self.category_freq_maps_: Dict[str, Dict[str, float]] = {}
        self.numeric_medians_: Dict[str, float] = {}
        self.scaler_means_: Dict[str, float] = {}
        self.scaler_scales_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.output_column_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit frequency maps, medians, and scalers strictly on training data."""
        clean_cols = [c for c in X.columns if c not in METADATA_AND_TARGET_COLS]
        self.feature_names_ = clean_cols
        self.categorical_cols_ = [
            c for c in clean_cols if c in ALL_RAW_CATEGORICAL_COLS or not pd.api.types.is_numeric_dtype(X[c])
        ]
        self.numeric_cols_ = [
            c for c in clean_cols if c not in self.categorical_cols_
        ]

        n_train = len(X)

        # 1. Train-fitted Relative Frequency Encoding for categoricals
        self.category_freq_maps_ = {}
        for col in self.categorical_cols_:
            counts = X[col].dropna().value_counts()
            freq_map = (counts / n_train).to_dict()
            self.category_freq_maps_[col] = freq_map

        # 2. Numeric medians
        self.numeric_medians_ = {}
        for col in self.numeric_cols_:
            vals = pd.to_numeric(X[col], errors="coerce")
            med = float(vals.median())
            self.numeric_medians_[col] = 0.0 if np.isnan(med) else med

        # 3. Fit StandardScaler on imputed numerics and frequency-encoded categoricals
        self.scaler_means_ = {}
        self.scaler_scales_ = {}

        for col in self.numeric_cols_:
            vals = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_medians_[col])
            mean = float(vals.mean())
            std = float(vals.std())
            self.scaler_means_[col] = mean
            self.scaler_scales_[col] = 1.0 if (std == 0.0 or np.isnan(std)) else std

        for col in self.categorical_cols_:
            freqs = X[col].map(self.category_freq_maps_[col]).fillna(0.0)
            mean = float(freqs.mean())
            std = float(freqs.std())
            self.scaler_means_[f"{col}_freq"] = mean
            self.scaler_scales_[f"{col}_freq"] = 1.0 if (std == 0.0 or np.isnan(std)) else std

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply train-fitted imputation, frequency encoding, unseen indicators, and scaling."""
        data_dict = {}

        # Numerics: scaled value + missing indicator (2 cols)
        for col in self.numeric_cols_:
            raw = pd.to_numeric(X[col], errors="coerce").values if col in X.columns else np.full(len(X), np.nan)
            is_missing = np.isnan(raw).astype(np.float32)
            imputed = np.where(np.isnan(raw), self.numeric_medians_[col], raw)
            scaled = (imputed - self.scaler_means_[col]) / self.scaler_scales_[col]

            data_dict[col] = scaled.astype(np.float32)
            data_dict[f"{col}_isna"] = is_missing

        # Categoricals: frequency scaled + is_unseen indicator + isna indicator (3 cols)
        for col in self.categorical_cols_:
            if col in X.columns:
                series = X[col]
                freq_map = self.category_freq_maps_[col]
                known_cats = set(freq_map.keys())

                # Unseen is where value is not null and not in train known categories
                is_unseen = series.notna() & (~series.isin(known_cats))
                is_missing = series.isna()

                # Map frequency, filling unseen and missing with 0.0
                freqs = series.map(freq_map).fillna(0.0).values
                scaled_freq = (freqs - self.scaler_means_[f"{col}_freq"]) / self.scaler_scales_[f"{col}_freq"]

                data_dict[f"{col}_freq"] = scaled_freq.astype(np.float32)
                data_dict[f"{col}_is_unseen"] = is_unseen.values.astype(np.float32)
                data_dict[f"{col}_isna"] = is_missing.values.astype(np.float32)
            else:
                data_dict[f"{col}_freq"] = np.zeros(len(X), dtype=np.float32)
                data_dict[f"{col}_is_unseen"] = np.ones(len(X), dtype=np.float32)
                data_dict[f"{col}_isna"] = np.ones(len(X), dtype=np.float32)

        df_out = pd.DataFrame(data_dict, index=X.index)
        self.output_column_names_ = list(df_out.columns)
        return df_out
