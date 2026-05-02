"""
tests/test_preprocessing.py
----------------------------
Unit tests for the preprocessing pipeline.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    rename_columns,
    drop_useless_columns,
    parse_dates,
    handle_missing_values,
    cap_outliers,
    encode_categoricals,
    add_derived_flags,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame that mimics the raw DataCo schema.
    Used as input for all preprocessing tests.
    """
    return pd.DataFrame({
        "order date (DateOrders)":       ["01/15/2017 10:00", "03/22/2017 14:30",
                                          "bad-date", None],
        "shipping date (DateOrders)":    ["01/20/2017 10:00", "03/28/2017 14:30",
                                          "01/25/2017 10:00", "04/01/2017 08:00"],
        "Order Id":                      [1001, 1002, 1003, 1004],
        "Order Item Quantity":           [5, 10, None, 3],
        "Sales":                         [100.0, 250.0, 75.0, None],
        "Order Profit Per Order":        [20.0, -5.0, 15.0, 8.0],
        "Benefit per order":             [20.0, -5.0, 15.0, 8.0],
        "Sales per customer":            [100.0, 125.0, 75.0, 80.0],
        "Product Name":                  ["Widget A", "Widget B", "Widget A", "Widget C"],
        "Category Name":                 ["Electronics", "Clothing", "Electronics", "Clothing"],
        "Order Region":                  ["North America", "Europe", "North America", "Asia"],
        "Market":                        ["US", "EU", "US", "APAC"],
        "Shipping Mode":                 ["Standard", "Express", "Standard", "Express"],
        "Order Status":                  ["COMPLETE", "CANCELED", "COMPLETE", "PENDING"],
        "Delivery Status":               ["Advance shipping", "Late delivery",
                                          "Shipping on time", "Late delivery"],
        "Late_delivery_risk":            [0, 1, 0, 1],
        "Days for shipping (real)":      [5, 6, 4, 7],
        "Days for shipment (scheduled)": [4, 4, 4, 4],
        "Order Item Product Price":      [20.0, 25.0, 15.0, 18.0],
        "Order Item Profit Ratio":       [0.2, -0.02, 0.2, 0.1],
        "Order Item Discount":           [0.0, 5.0, 0.0, 2.0],
        "Order Item Discount Rate":      [0.0, 0.02, 0.0, 0.01],
        "Order Item Total":              [100.0, 245.0, 75.0, 54.0],
        "Type":                          ["DEBIT", "CREDIT", "DEBIT", "CASH"],
        "Customer Segment":              ["Consumer", "Corporate", "Consumer", "Home Office"],
        "Department Name":               ["Electronics", "Apparel", "Electronics", "Apparel"],
        "Customer City":                 ["New York", "London", "New York", "Tokyo"],
        "Customer Country":              ["US", "UK", "US", "JP"],
        "Customer Id":                   [501, 502, 503, 504],
        "Customer State":                ["NY", "ENG", "NY", "TKY"],
        "Customer Zipcode":              ["10001", "EC1A", "10001", "100-0001"],
        "Department Id":                 [1, 2, 1, 2],
        "Latitude":                      [40.7, 51.5, 40.7, 35.6],
        "Longitude":                     [-74.0, -0.1, -74.0, 139.6],
        "Order City":                    ["New York", "London", "New York", "Tokyo"],
        "Order Country":                 ["US", "UK", "US", "JP"],
        "Order Customer Id":             [501, 502, 503, 504],
        "Order Item Cardprod Id":        [201, 202, 201, 203],
        "Order Item Id":                 [301, 302, 303, 304],
        "Order Region":                  ["North America", "Europe", "North America", "Asia"],
        "Order State":                   ["NY", "ENG", "NY", "TKY"],
        "Order Zipcode":                 ["10001", "EC1A", "10001", "100-0001"],
        "Product Card Id":               [201, 202, 201, 203],
        "Product Category Id":           [1, 2, 1, 2],
        "Product Price":                 [20.0, 25.0, 15.0, 18.0],
        "Product Status":                [0, 0, 0, 0],
        "Category Id":                   [1, 2, 1, 2],
        # PII columns that should be dropped
        "Customer Email":                ["a@b.com", "c@d.com", "e@f.com", "g@h.com"],
        "Customer Fname":                ["Alice", "Bob", "Carol", "Dave"],
        "Customer Lname":                ["Smith", "Jones", "Brown", "White"],
        "Customer Password":             ["pass1", "pass2", "pass3", "pass4"],
        "Customer Street":               ["123 Main St", "456 High St", "789 Oak Ave", "321 Pine Rd"],
        "Product Image":                 ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        "Product Description":           ["Desc 1", "Desc 2", "Desc 3", "Desc 4"],
    })


# ── Tests: rename_columns ─────────────────────────────────────────────────────

class TestRenameColumns:
    def test_order_date_renamed(self, raw_sample):
        df = rename_columns(raw_sample)
        assert "order_date" in df.columns

    def test_original_column_gone(self, raw_sample):
        df = rename_columns(raw_sample)
        assert "order date (DateOrders)" not in df.columns

    def test_order_quantity_renamed(self, raw_sample):
        df = rename_columns(raw_sample)
        assert "order_quantity" in df.columns

    def test_product_name_renamed(self, raw_sample):
        df = rename_columns(raw_sample)
        assert "product_name" in df.columns


# ── Tests: drop_useless_columns ───────────────────────────────────────────────

class TestDropUselessColumns:
    def test_pii_columns_dropped(self, raw_sample):
        df = rename_columns(raw_sample)
        df = drop_useless_columns(df)
        pii = ["Customer Email", "Customer Fname", "Customer Lname",
               "Customer Password", "Customer Street", "Product Image"]
        for col in pii:
            assert col not in df.columns, f"PII column still present: {col}"

    def test_core_columns_retained(self, raw_sample):
        df = rename_columns(raw_sample)
        df = drop_useless_columns(df)
        essential = ["order_quantity", "sales", "product_name", "category_name"]
        for col in essential:
            assert col in df.columns, f"Essential column dropped: {col}"

    def test_row_count_unchanged(self, raw_sample):
        df = rename_columns(raw_sample)
        df = drop_useless_columns(df)
        assert len(df) == len(raw_sample)


# ── Tests: parse_dates ────────────────────────────────────────────────────────

class TestParseDates:
    def test_order_date_is_datetime(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(df["order_date"])

    def test_bad_dates_become_nat(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        nat_count = df["order_date"].isna().sum()
        assert nat_count >= 1  # "bad-date" and None should be NaT

    def test_derived_date_columns_created(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        for col in ["order_year", "order_month", "order_week", "order_dow"]:
            assert col in df.columns, f"Missing derived column: {col}"

    def test_year_values_correct(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        valid = df["order_year"].dropna()
        assert all(valid == 2017)

    def test_actual_lead_time_calculated(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        assert "actual_lead_time" in df.columns


# ── Tests: handle_missing_values ─────────────────────────────────────────────

class TestHandleMissingValues:
    def test_nat_rows_dropped(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        before = len(df)
        df = handle_missing_values(df)
        # Rows with NaT order_date should be removed
        assert len(df) < before

    def test_no_nulls_in_numeric_cols(self, raw_sample):
        df = rename_columns(raw_sample)
        df = drop_useless_columns(df)
        df = parse_dates(df)
        df = handle_missing_values(df)
        numeric_nulls = df.select_dtypes(include=[np.number]).isna().sum().sum()
        assert numeric_nulls == 0

    def test_no_nulls_in_string_cols(self, raw_sample):
        df = rename_columns(raw_sample)
        df = drop_useless_columns(df)
        df = parse_dates(df)
        df = handle_missing_values(df)
        str_nulls = df.select_dtypes(include=["object"]).isna().sum().sum()
        assert str_nulls == 0


# ── Tests: cap_outliers ───────────────────────────────────────────────────────

class TestCapOutliers:
    def test_outliers_capped(self):
        df = pd.DataFrame({
            "sales": [10, 12, 11, 13, 10000],  # 10000 is extreme outlier
        })
        df_capped = cap_outliers(df, cols=["sales"], iqr_factor=1.5)
        assert df_capped["sales"].max() < 10000

    def test_normal_values_unchanged(self):
        df = pd.DataFrame({
            "sales": [10, 12, 11, 13, 14],  # no outliers
        })
        original_max = df["sales"].max()
        df_capped = cap_outliers(df, cols=["sales"], iqr_factor=1.5)
        assert df_capped["sales"].max() == original_max

    def test_missing_columns_ignored(self):
        df = pd.DataFrame({"sales": [10, 12, 11]})
        # Should not raise even if column doesn't exist
        df_capped = cap_outliers(df, cols=["sales", "nonexistent_col"])
        assert "nonexistent_col" not in df_capped.columns


# ── Tests: add_derived_flags ──────────────────────────────────────────────────

class TestDerivedFlags:
    def test_is_late_flag_created(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        df = handle_missing_values(df)
        df = add_derived_flags(df)
        assert "is_late" in df.columns

    def test_is_late_is_binary(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        df = handle_missing_values(df)
        df = add_derived_flags(df)
        assert set(df["is_late"].unique()).issubset({0, 1})

    def test_is_cancelled_flag_created(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        df = handle_missing_values(df)
        df = add_derived_flags(df)
        assert "is_cancelled" in df.columns

    def test_cancelled_orders_flagged_correctly(self, raw_sample):
        df = rename_columns(raw_sample)
        df = parse_dates(df)
        df = handle_missing_values(df)
        df = add_derived_flags(df)
        cancelled = df[df["order_status"] == "CANCELED"]["is_cancelled"]
        assert all(cancelled == 1)


# ── Tests: encode_categoricals ───────────────────────────────────────────────

class TestEncodeCategories:
    def test_encoded_columns_created(self, raw_sample):
        df = rename_columns(raw_sample)
        df = drop_useless_columns(df)
        df = parse_dates(df)
        df = handle_missing_values(df)
        df = encode_categoricals(df, cols=["category_name", "order_region"])
        assert "category_name_enc" in df.columns
        assert "order_region_enc" in df.columns

    def test_encoded_values_are_integers(self, raw_sample):
        df = rename_columns(raw_sample)
        df = drop_useless_columns(df)
        df = parse_dates(df)
        df = handle_missing_values(df)
        df = encode_categoricals(df, cols=["category_name"])
        assert pd.api.types.is_integer_dtype(df["category_name_enc"])

    def test_original_column_retained(self, raw_sample):
        df = rename_columns(raw_sample)
        df = drop_useless_columns(df)
        df = parse_dates(df)
        df = handle_missing_values(df)
        df = encode_categoricals(df, cols=["category_name"])
        assert "category_name" in df.columns


# ── Tests: full pipeline ──────────────────────────────────────────────────────

class TestFullPipeline:
    def test_preprocess_runs_without_error(self, raw_sample):
        from src.preprocessing import preprocess
        df = preprocess(raw_sample)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_output_has_no_nulls_in_numeric(self, raw_sample):
        from src.preprocessing import preprocess
        df = preprocess(raw_sample)
        numeric_nulls = df.select_dtypes(include=[np.number]).isna().sum().sum()
        assert numeric_nulls == 0

    def test_output_shape_reasonable(self, raw_sample):
        from src.preprocessing import preprocess
        df = preprocess(raw_sample)
        # Should have more columns than input (derived features added)
        assert df.shape[1] > 10
        # Should have fewer rows (NaT dates dropped)
        assert df.shape[0] <= len(raw_sample)
