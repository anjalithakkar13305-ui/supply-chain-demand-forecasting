"""
tests/test_feature_engineering.py
----------------------------------
Unit tests for the feature engineering pipeline.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import (
    aggregate_weekly,
    add_lag_features,
    add_rolling_features,
    add_seasonality_features,
    add_lead_time_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_sample() -> pd.DataFrame:
    """
    Minimal preprocessed DataFrame for feature engineering tests.
    Has 3 products × 16 weeks = 48 rows to allow lag feature creation.
    """
    products = ["Widget A", "Widget B", "Widget C"]
    weeks = pd.date_range("2016-01-04", periods=16, freq="W")
    rows = []
    for product in products:
        for week in weeks:
            rows.append({
                "order_date":              week,
                "order_quantity":          np.random.randint(5, 50),
                "sales":                   np.random.uniform(100, 500),
                "order_profit":            np.random.uniform(10, 100),
                "category_name":           "Electronics",
                "order_region":            "North America",
                "market":                  "US",
                "product_name":            product,
                "item_discount_rate":      np.random.uniform(0, 0.1),
                "days_shipping_real":      np.random.randint(3, 8),
                "days_shipping_scheduled": 4,
                "is_late":                 np.random.randint(0, 2),
                "order_id":                np.random.randint(1000, 9999),
            })
    return pd.DataFrame(rows)


# ── Tests: aggregate_weekly ───────────────────────────────────────────────────

class TestAggregateWeekly:
    def test_returns_dataframe(self, clean_sample):
        result = aggregate_weekly(clean_sample)
        assert isinstance(result, pd.DataFrame)

    def test_has_demand_column(self, clean_sample):
        result = aggregate_weekly(clean_sample)
        assert "demand" in result.columns

    def test_has_week_start_column(self, clean_sample):
        result = aggregate_weekly(clean_sample)
        assert "week_start" in result.columns

    def test_demand_is_non_negative(self, clean_sample):
        result = aggregate_weekly(clean_sample)
        assert (result["demand"] >= 0).all()

    def test_grouped_by_product(self, clean_sample):
        result = aggregate_weekly(clean_sample)
        assert result["product_name"].nunique() == 3


# ── Tests: add_lag_features ───────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag_columns_created(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_lag_features(weekly, lags=[1, 2, 4])
        assert "demand_lag_1w" in result.columns
        assert "demand_lag_2w" in result.columns
        assert "demand_lag_4w" in result.columns

    def test_lag_1_is_previous_week(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_lag_features(weekly, lags=[1])
        # For a single product, lag_1w should equal the previous row's demand
        prod_a = result[result["product_name"] == "Widget A"].sort_values("week_start")
        # Row index 1: lag_1w should equal row index 0's demand
        lag_val    = prod_a["demand_lag_1w"].iloc[1]
        actual_val = prod_a["demand"].iloc[0]
        assert lag_val == actual_val

    def test_first_rows_are_nan(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_lag_features(weekly, lags=[1])
        prod_a = result[result["product_name"] == "Widget A"].sort_values("week_start")
        assert pd.isna(prod_a["demand_lag_1w"].iloc[0])

    def test_no_data_leakage_across_products(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_lag_features(weekly, lags=[1])
        # First row of Widget B should have NaN lag, not Widget A's last value
        prod_b_first = result[result["product_name"] == "Widget B"].sort_values("week_start").iloc[0]
        assert pd.isna(prod_b_first["demand_lag_1w"])


# ── Tests: add_rolling_features ───────────────────────────────────────────────

class TestRollingFeatures:
    def test_rolling_columns_created(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_rolling_features(weekly, windows=[4])
        assert "demand_roll_mean_4w" in result.columns
        assert "demand_roll_std_4w" in result.columns

    def test_rolling_mean_non_negative(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_rolling_features(weekly, windows=[4])
        # NaN at product boundaries is expected (first row per product)
        non_null = result["demand_roll_mean_4w"].dropna()
        assert (non_null >= 0).all()

    def test_rolling_std_non_negative(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_rolling_features(weekly, windows=[4])
        assert (result["demand_roll_std_4w"] >= 0).all()

    def test_no_nulls_due_to_min_periods(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_rolling_features(weekly, windows=[4])
        # First row per product group is NaN (shift(1) makes it null)
        # Null count should equal the number of unique products (one per group)
        n_products = result["product_name"].nunique()
        assert result["demand_roll_mean_4w"].isna().sum() == n_products


# ── Tests: add_seasonality_features ──────────────────────────────────────────

class TestSeasonalityFeatures:
    def test_all_seasonality_columns_created(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_seasonality_features(weekly)
        expected = ["month_sin", "month_cos", "week_of_year_sin",
                    "week_of_year_cos", "quarter", "is_q4",
                    "is_year_end", "is_summer", "is_holiday_season"]
        for col in expected:
            assert col in result.columns, f"Missing seasonality column: {col}"

    def test_sin_cos_in_range(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_seasonality_features(weekly)
        assert result["month_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()

    def test_quarter_values_valid(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_seasonality_features(weekly)
        assert set(result["quarter"].unique()).issubset({1, 2, 3, 4})

    def test_binary_flags_are_binary(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_seasonality_features(weekly)
        for col in ["is_q4", "is_year_end", "is_summer", "is_holiday_season"]:
            assert set(result[col].unique()).issubset({0, 1}), \
                f"{col} contains non-binary values"


# ── Tests: add_lead_time_features ─────────────────────────────────────────────

class TestLeadTimeFeatures:
    def test_shipping_delay_calculated(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_lead_time_features(weekly)
        assert "shipping_delay" in result.columns

    def test_delay_ratio_calculated(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_lead_time_features(weekly)
        assert "delay_ratio" in result.columns

    def test_shipping_delay_is_numeric(self, clean_sample):
        weekly = aggregate_weekly(clean_sample)
        result = add_lead_time_features(weekly)
        assert pd.api.types.is_numeric_dtype(result["shipping_delay"])
