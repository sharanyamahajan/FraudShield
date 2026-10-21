"""Feature extraction and preprocessing pipelines for Feature Sets A and B.

Set A: Raw transaction attributes + simple transforms (log amount, cents, approximate hour).
        Strictly excludes any V, C, D, M, or id_* columns.
Set B: Set A + provided anonymised C, D, M, V columns and identity table columns.

Every encoder, imputer, and scaler is fitted STRICTLY on the TRAIN split.
Unseen categories in validation or test splits are mapped to a reserved value (-1).
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("fraudshield.features.pipeline")

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
    ]
    if forbidden:
        raise ValueError(f"Set A contains forbidden columns: {forbidden}")

    return df_out


def get_set_b_additional_columns(available_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Identify available C, D, M, V, and identity columns for Set B."""
    b_numeric = []
    b_categorical = []

    for col in available_cols:
        # Check C, D, V columns (mostly numeric anonymised)
        if any(col.startswith(p) for p in ["C", "D", "V"]) and col not in ["DeviceInfo", "DeviceType"]:
            b_numeric.append(col)
        # Check M columns (anonymised match flags, e.g. M1..M9)
        elif col.startswith("M") and len(col) <= 3 and col[1:].isdigit():
            b_categorical.append(col)
        # Check Identity columns
        elif col.startswith("id_"):
            # id_01 to id_11 are typically numeric, id_12 to id_38 are mix of cat and num
            # We treat id_12..id_38 as categorical if object, else numeric
            # We'll determine dynamically based on dtype or standard specification
            b_numeric.append(col)
        elif col in ["DeviceType", "DeviceInfo"]:
            b_categorical.append(col)

    return b_numeric, b_categorical


def extract_set_b(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    """Extract Set B features: Set A + provided C, D, M, V and identity columns."""
    df_a = extract_set_a_raw(df, time_col=time_col)
    df_b = df_a.copy()

    # Identify all Set B eligible columns present in df
    set_b_cols = [
        col for col in df.columns
        if col not in df_a.columns
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
        self.feature_names_ = list(X.columns)
        self.categorical_cols_ = [
            c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
        ]
        self.numeric_cols_ = [
            c for c in X.columns if c not in self.categorical_cols_
        ]

        self.category_maps_ = {}
        for col in self.categorical_cols_:
            # Extract distinct non-null categories from train
            valid_cats = X[col].dropna().unique()
            # Map each known training category to an integer starting from 0
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

        # Encode categoricals
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
    
    Fits median imputers, missing indicators, and standard scalers STRICTLY on the train split.
    Unseen categories in validation/test are mapped to -1.
    """

    def __init__(self):
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.numeric_medians_: Dict[str, float] = {}
        self.scaler_means_: Dict[str, float] = {}
        self.scaler_scales_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers and scalers strictly on training data."""
        self.feature_names_ = list(X.columns)
        self.categorical_cols_ = [
            c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
        ]
        self.numeric_cols_ = [
            c for c in X.columns if c not in self.categorical_cols_
        ]

        # 1. Categorical maps
        self.category_maps_ = {}
        for col in self.categorical_cols_:
            valid_cats = X[col].dropna().unique()
            self.category_maps_[col] = {cat: i for i, cat in enumerate(sorted(valid_cats))}

        # 2. Numeric medians
        self.numeric_medians_ = {}
        for col in self.numeric_cols_:
            vals = pd.to_numeric(X[col], errors="coerce")
            med = float(vals.median())
            self.numeric_medians_[col] = 0.0 if np.isnan(med) else med

        # Compute imputed values to fit scaler
        imputed_numeric = pd.DataFrame(index=X.index)
        for col in self.numeric_cols_:
            vals = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_medians_[col])
            imputed_numeric[col] = vals

        self.scaler_means_ = {}
        self.scaler_scales_ = {}
        for col in self.numeric_cols_:
            mean = float(imputed_numeric[col].mean())
            std = float(imputed_numeric[col].std())
            self.scaler_means_[col] = mean
            self.scaler_scales_[col] = 1.0 if (std == 0 or np.isnan(std)) else std

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply train-fitted imputation, missing indicators, and scaling."""
        X_out = pd.DataFrame(index=X.index)

        # Process numerics: imputed + scaled + missing indicator
        for col in self.numeric_cols_:
            raw = pd.to_numeric(X[col], errors="coerce") if col in X.columns else pd.Series(np.nan, index=X.index)
            is_missing = raw.isna().astype(np.float32)
            imputed = raw.fillna(self.numeric_medians_[col])
            scaled = (imputed - self.scaler_means_[col]) / self.scaler_scales_[col]

            X_out[col] = scaled.astype(np.float32)
            X_out[f"{col}_isna"] = is_missing

        # Process categoricals: mapped + missing indicator
        for col in self.categorical_cols_:
            if col in X.columns:
                cat_map = self.category_maps_[col]
                mapped = X[col].map(cat_map).fillna(-1)
                mapped = np.where(X[col].isna(), -2, mapped)
                X_out[col] = mapped.astype(np.float32)
                X_out[f"{col}_isna"] = X[col].isna().astype(np.float32)
            else:
                X_out[col] = -2.0
                X_out[f"{col}_isna"] = 1.0

        return X_out
