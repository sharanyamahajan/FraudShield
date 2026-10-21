"""Unit tests for feature pipelines (Sets A and B) and train-only preprocessing."""

import pytest
import numpy as np
import pandas as pd
from fraudshield.features.pipeline import (
    extract_set_a_raw,
    extract_set_b,
    TreePreprocessor,
    LinearPreprocessor,
)


@pytest.fixture
def mock_transaction_dataframe():
    """Create a mock transaction dataframe containing raw, anonymised, and identity columns."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [1000, 2000, 3000, 4000, 5000],
        "TransactionAmt": [50.0, 100.25, 0.99, 500.0, 25.5],
        "ProductCD": ["W", "H", "W", "C", "R"],
        "card1": [1001, 1002, 1001, 1003, 1002],
        "card2": [100.0, 200.0, np.nan, 100.0, 300.0],
        "card3": [150.0, 150.0, 150.0, np.nan, 150.0],
        "card4": ["visa", "mastercard", "visa", "discover", "visa"],
        "card5": [226.0, 137.0, 226.0, np.nan, 226.0],
        "card6": ["debit", "credit", "debit", "credit", "debit"],
        "addr1": [315.0, np.nan, 315.0, 420.0, 315.0],
        "addr2": [87.0, 87.0, 87.0, 87.0, 87.0],
        "dist1": [10.0, np.nan, 5.0, np.nan, 12.0],
        "dist2": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "P_emaildomain": ["gmail.com", "yahoo.com", "gmail.com", "anonymous.info", np.nan],
        "R_emaildomain": [np.nan, "gmail.com", np.nan, np.nan, "hotmail.com"],
        # Anonymised and identity columns
        "C1": [1.0, 2.0, 1.0, 1.0, 3.0],
        "D1": [14.0, 0.0, 10.0, np.nan, 5.0],
        "M1": ["T", "F", "T", np.nan, "T"],
        "V1": [1.0, 1.0, np.nan, 1.0, 0.0],
        "DeviceType": ["mobile", "desktop", np.nan, "mobile", np.nan],
        "DeviceInfo": ["iOS", "Windows", np.nan, "Android", np.nan],
        "id_01": [-5.0, 0.0, np.nan, -10.0, np.nan],
        "isFraud": [0, 0, 1, 0, 0],
    })


def test_set_a_strictly_excludes_anonymised_and_identity_columns(mock_transaction_dataframe):
    """Assert Set A contains only raw transaction fields and strictly excludes V, C, D, M, id, and Device columns."""
    df_a = extract_set_a_raw(mock_transaction_dataframe)

    # Expected raw and engineered features
    expected_cols = {
        "TransactionAmt", "card1", "card2", "card3", "card5",
        "addr1", "addr2", "dist1", "dist2",
        "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
        "log_amount", "amount_cents", "hour_of_day_proxy",
    }
    assert set(df_a.columns) == expected_cols

    # Assert strictly no forbidden columns
    forbidden = [
        col for col in df_a.columns
        if any(col.startswith(p) for p in ["V", "C", "D", "M", "id_"])
        or col in ["DeviceType", "DeviceInfo"]
    ]
    assert len(forbidden) == 0, f"Found forbidden columns in Set A: {forbidden}"


def test_set_b_includes_set_a_and_provided_columns(mock_transaction_dataframe):
    """Assert Set B includes all Set A features plus anonymised C, D, M, V and identity columns."""
    df_a = extract_set_a_raw(mock_transaction_dataframe)
    df_b = extract_set_b(mock_transaction_dataframe)

    assert set(df_a.columns).issubset(set(df_b.columns))
    assert "C1" in df_b.columns
    assert "D1" in df_b.columns
    assert "M1" in df_b.columns
    assert "V1" in df_b.columns
    assert "DeviceType" in df_b.columns
    assert "DeviceInfo" in df_b.columns
    assert "id_01" in df_b.columns


def test_train_only_fitting_tree_preprocessor_is_independent_of_val_test():
    """Verify that TreePreprocessor fitted on Train only has identical state whether or not val/test is seen."""
    train_df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0],
        "cat1": ["A", "B", "A", "C"],
    })
    val_test_df = pd.DataFrame({
        "num1": [10.0, 20.0],
        "cat1": ["B", "NEW_CATEGORY"],  # Contains unseen category
    })

    # Fit transformer 1 on Train only
    prep1 = TreePreprocessor()
    prep1.fit(train_df)

    # Fit transformer 2 on Train only (while completely ignoring val_test_df)
    prep2 = TreePreprocessor()
    prep2.fit(train_df)

    assert prep1.category_maps_ == prep2.category_maps_
    assert prep1.numeric_cols_ == prep2.numeric_cols_
    assert prep1.categorical_cols_ == prep2.categorical_cols_

    # Verify that unseen category in val/test maps to -1
    transformed_val = prep1.transform(val_test_df)
    assert transformed_val["cat1"].iloc[0] == prep1.category_maps_["cat1"]["B"]
    assert transformed_val["cat1"].iloc[1] == -1.0  # NEW_CATEGORY mapped to -1


def test_train_only_fitting_linear_preprocessor_is_independent_of_val_test():
    """Verify that LinearPreprocessor fitted on Train only is completely uninfluenced by val/test distributions."""
    train_df = pd.DataFrame({
        "num1": [10.0, 20.0, 30.0, np.nan],  # median = 20.0
        "cat1": ["low", "high", "low", "medium"],
    })
    val_df = pd.DataFrame({
        "num1": [1000.0, 2000.0],  # Massive outlier distribution in val
        "cat1": ["high", "unseen_cat"],
    })

    prep = LinearPreprocessor()
    prep.fit(train_df)

    # Assert training median was learned from train only
    assert prep.numeric_medians_["num1"] == 20.0
    assert "unseen_cat" not in prep.category_freq_maps_["cat1"]

    # Transform val
    transformed_val = prep.transform(val_df)
    # Check that unseen category maps to 0.0 frequency and is_unseen indicator is 1.0
    assert transformed_val["cat1_is_unseen"].iloc[1] == 1.0
    assert transformed_val["cat1_freq"].iloc[1] < 0.0  # (0.0 - mean) / std is negative
    # Check that missing indicator exists
    assert "num1_isna" in transformed_val.columns


def test_metadata_and_target_strictly_excluded_from_model_features(mock_transaction_dataframe):
    """Assert TransactionID, TransactionDT, and isFraud are strictly excluded from model feature sets."""
    from fraudshield.features.pipeline import get_model_feature_columns, METADATA_AND_TARGET_COLS

    df_a = extract_set_a_raw(mock_transaction_dataframe)
    df_b = extract_set_b(mock_transaction_dataframe)

    cols_a = get_model_feature_columns(df_a)
    cols_b = get_model_feature_columns(df_b)

    for forbidden in METADATA_AND_TARGET_COLS:
        assert forbidden not in cols_a, f"{forbidden} found in Set A model feature columns!"
        assert forbidden not in cols_b, f"{forbidden} found in Set B model feature columns!"


def test_unseen_categories_map_to_reserved_value():
    """Verify that unseen categories in val/test are mapped to designated reserved values."""
    train_df = pd.DataFrame({"cat": ["red", "blue", "blue"]})
    test_df = pd.DataFrame({"cat": ["red", "green", np.nan]})  # 'green' is unseen

    # TreePreprocessor maps unseen -> -1, missing -> -2
    tree_prep = TreePreprocessor().fit(train_df)
    tree_res = tree_prep.transform(test_df)
    assert tree_res["cat"].iloc[0] >= 0  # known category
    assert tree_res["cat"].iloc[1] == -1.0  # unseen category 'green' mapped to -1
    assert tree_res["cat"].iloc[2] == -2.0  # missing category mapped to -2

    # LinearPreprocessor maps unseen -> frequency 0.0 and _is_unseen = 1.0
    linear_prep = LinearPreprocessor().fit(train_df)
    linear_res = linear_prep.transform(test_df)
    assert linear_res["cat_is_unseen"].iloc[0] == 0.0  # known
    assert linear_res["cat_is_unseen"].iloc[1] == 1.0  # unseen 'green'
    assert linear_res["cat_is_unseen"].iloc[2] == 0.0  # missing is not unseen, it's missing
    assert linear_res["cat_isna"].iloc[2] == 1.0       # missing flag

