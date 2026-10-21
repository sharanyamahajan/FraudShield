"""Feature extraction and preprocessing pipelines for Feature Sets A and B.

Set A: Raw transaction attributes + simple transforms (log amount, cents, approximate hour).
        Strictly excludes any V, C, D, M, or id_* columns.
Set B: Set A + provided anonymised C, D, M, V columns and identity table columns.

Every encoder, imputer, and scaler is fitted STRICTLY on the TRAIN split.
- Tree models: Ordinal/integer encoding on Train; unseen -> -1, missing -> -2; native NaN handling.
- Linear models: Train-fitted relative frequency encoding for categoricals with explicit
  unseen token (0.0 frequency) and _is_unseen indicator; median imputation and standard scaling.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("fraudshield.features.pipeline")

# Columns that must NEVER be fed into ML models as feature inputs
METADATA_AND_TARGET_COLS = ["TransactionID", "TransactionDT", "isFraud"]

# Raw transaction columns forming Set A
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

# Simple engineered transforms in Set A
SET_A_ENGINEERED = [
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
    """
    df_out = pd.DataFrame(index=df.index)

    # 1. Base numeric columns
    for col in SET_A_RAW_NUMERIC:
        if col in df.columns:
            df_out[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        else:
            df_out[col] = np.nan

    # 2. Base categorical columns
    for col in SET_A_RAW_CATEGORICAL:
        if col in df.columns:
            df_out[col] = df[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
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


def extract_set_b(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    """Extract Set B features: Set A + provided C, D, M, V and identity columns."""
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
        if pd.api.types.is_numeric_dtype(df[col]):
            df_b[col] = df[col].astype(np.float32)
        else:
            df_b[col] = df[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})

    return df_b


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
            c for c in clean_cols if not pd.api.types.is_numeric_dtype(X[c])
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
    
    Categorical handling:
    - Never uses arbitrary integer codes.
    - Fits relative frequency encoding strictly on TRAIN: freq(cat) = count(cat) / N_train.
    - Unseen categories in validation/test receive frequency = 0.0 and an explicit _is_unseen indicator.
    - Missing categorical values receive frequency = 0.0 and an explicit _isna indicator.
    - Relative frequencies are scaled using train-fitted mean and standard deviation.

    Numeric handling:
    - Median imputation fitted strictly on TRAIN.
    - Explicit _isna missingness indicator.
    - StandardScaler fitted strictly on imputed TRAIN features.
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
            c for c in clean_cols if not pd.api.types.is_numeric_dtype(X[c])
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

        # Numerics: scaled value + missing indicator
        for col in self.numeric_cols_:
            raw = pd.to_numeric(X[col], errors="coerce").values if col in X.columns else np.full(len(X), np.nan)
            is_missing = np.isnan(raw).astype(np.float32)
            imputed = np.where(np.isnan(raw), self.numeric_medians_[col], raw)
            scaled = (imputed - self.scaler_means_[col]) / self.scaler_scales_[col]

            data_dict[col] = scaled.astype(np.float32)
            data_dict[f"{col}_isna"] = is_missing

        # Categoricals: frequency scaled + is_unseen indicator + isna indicator
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
