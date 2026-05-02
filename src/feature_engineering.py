"""
feature_engineering.py
-----------------------
Builds the feature matrix for demand forecasting.

Core transformations
--------------------
1. Aggregate raw order rows to a weekly time-series (product × week).
2. Create lag features (demand 1, 2, 4, 8 weeks ago).
3. Create rolling-window statistics (mean, std over 4-week and 8-week windows).
4. Add seasonality flags (month, quarter, holiday proximity, year-end surge).
5. Add lead-time features derived from scheduled vs actual shipping days.
6. Encode external regressors (market, region, category) as integers.

Output
------
Two DataFrames are returned:
- `feature_df` : full feature matrix for ML models (XGBoost etc.)
- `prophet_df` : ds/y/regressor format for Prophet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lag periods in weeks
LAG_PERIODS = [1, 2, 4, 8, 12]

# Rolling window sizes in weeks
ROLLING_WINDOWS = [4, 8, 12]


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_weekly(df: pd.DataFrame,
                     group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Aggregate order-level data to weekly demand per product+region+category.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from preprocessing.preprocess().
    group_cols : list[str] or None
        Columns to group by in addition to week. Defaults to
        ['product_name', 'category_name', 'order_region', 'market'].

    Returns
    -------
    pd.DataFrame
        One row per (week_start, *group_cols) with aggregated metrics.
    """
    if group_cols is None:
        group_cols = ["product_name", "category_name", "order_region", "market"]

    # Snap order_date to the Monday of each ISO week
    df = df.copy()
    df["week_start"] = df["order_date"].dt.to_period("W").dt.start_time

    agg_funcs = {
        "order_quantity": "sum",
        "sales":          "sum",
        "order_profit":   "sum",
        "item_discount_rate": "mean",
        "days_shipping_real": "mean",
        "days_shipping_scheduled": "mean",
        "is_late":        "mean",
        "order_id":       "count",   # acts as order_count
    }
    # Keep only columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}

    weekly = (
        df.groupby(["week_start"] + group_cols)
          .agg(agg_funcs)
          .reset_index()
    )
    if "order_id" in weekly.columns:
        weekly = weekly.rename(columns={"order_id": "order_count"})

    # Rename target
    weekly = weekly.rename(columns={"order_quantity": "demand"})

    weekly = weekly.sort_values(["product_name", "week_start"]).reset_index(drop=True)
    logger.info("Weekly aggregation: %d rows, %d unique products",
                len(weekly), weekly["product_name"].nunique())
    return weekly


# ── Lag features ───────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     target_col: str = "demand",
                     group_col: str = "product_name",
                     lags: list[int] = LAG_PERIODS) -> pd.DataFrame:
    """
    Add lag features for the target column within each group.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly aggregated DataFrame sorted by (group_col, week_start).
    target_col : str
        Column to create lags for.
    group_col : str
        The grouping key (usually product or category).
    lags : list[int]
        Number of periods to lag.

    Returns
    -------
    pd.DataFrame with new lag_N columns.
    """
    df = df.sort_values([group_col, "week_start"]).copy()
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}w"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        logger.debug("Added lag feature: %s", col_name)
    return df


# ── Rolling statistics ─────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame,
                          target_col: str = "demand",
                          group_col: str = "product_name",
                          windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features.

    Uses min_periods=1 so early rows don't become NaN.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    group_col : str
    windows : list[int]
        Window sizes in weeks.

    Returns
    -------
    pd.DataFrame
    """
    df = df.sort_values([group_col, "week_start"]).copy()
    for w in windows:
        mean_col = f"{target_col}_roll_mean_{w}w"
        std_col  = f"{target_col}_roll_std_{w}w"
        grouped  = df.groupby(group_col)[target_col]
        df[mean_col] = grouped.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[std_col] = grouped.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std().fillna(0)
        )
        logger.debug("Added rolling features: %s, %s", mean_col, std_col)
    return df


# ── Seasonality flags ─────────────────────────────────────────────────────────

def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based seasonality indicators.

    Columns added
    -------------
    - month_sin / month_cos : cyclic encoding of month
    - week_of_year_sin / _cos : cyclic encoding of ISO week
    - quarter : 1–4
    - is_q4 : boolean flag for October–December (holiday surge)
    - is_year_end : November + December
    - is_summer : June–August
    - is_weekend_heavy : week containing a US long-weekend
    """
    df = df.copy()
    week_dt = pd.to_datetime(df["week_start"])

    month = week_dt.dt.month
    week_num = week_dt.dt.isocalendar().week.astype(float)

    df["month_sin"]        = np.sin(2 * np.pi * month / 12)
    df["month_cos"]        = np.cos(2 * np.pi * month / 12)
    df["week_of_year_sin"] = np.sin(2 * np.pi * week_num / 52)
    df["week_of_year_cos"] = np.cos(2 * np.pi * week_num / 52)
    df["quarter"]          = week_dt.dt.quarter
    df["is_q4"]            = (month >= 10).astype(int)
    df["is_year_end"]      = (month >= 11).astype(int)
    df["is_summer"]        = month.isin([6, 7, 8]).astype(int)
    df["week_of_year"]     = week_num.astype(int)
    df["year"]             = week_dt.dt.year

    # Simple US holiday proximity flag: weeks 47–52 (Thanksgiving/Christmas)
    df["is_holiday_season"] = week_num.between(47, 52).astype(int)

    logger.info("Seasonality features added.")
    return df


# ── Lead-time features ────────────────────────────────────────────────────────

def add_lead_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive features that capture supply-side constraints.

    - shipping_delay : actual − scheduled days (positive = late)
    - delay_ratio    : shipping_delay / scheduled_days
    - is_chronically_late : rolling mean of is_late > 0.5 (lagged by 1 period)
    """
    df = df.copy()

    if "days_shipping_real" in df.columns and "days_shipping_scheduled" in df.columns:
        df["shipping_delay"] = df["days_shipping_real"] - df["days_shipping_scheduled"]
        df["delay_ratio"] = np.where(
            df["days_shipping_scheduled"] > 0,
            df["shipping_delay"] / df["days_shipping_scheduled"],
            0,
        )

    if "is_late" in df.columns:
        df["is_chronically_late"] = (
            df.groupby("product_name")["is_late"]
              .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean() > 0.5)
              .astype(int)
        )

    logger.info("Lead-time features added.")
    return df


# ── Encode group columns ──────────────────────────────────────────────────────

def encode_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Integer-encode high-cardinality group columns for use as XGBoost features.

    Produces: product_name_enc, category_name_enc, order_region_enc, market_enc
    """
    df = df.copy()
    for col in ["product_name", "category_name", "order_region", "market"]:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].astype("category").cat.codes
    logger.info("Group columns encoded.")
    return df


# ── Prophet format ────────────────────────────────────────────────────────────

def build_prophet_df(weekly_df: pd.DataFrame,
                     product_name: str | None = None) -> pd.DataFrame:
    """
    Convert weekly aggregated data to Prophet's ds/y format.

    If product_name is given, filter to that product only and return a
    simple (ds, y) DataFrame.  If None, aggregate across all products
    (total market demand).

    Parameters
    ----------
    weekly_df : pd.DataFrame
    product_name : str or None

    Returns
    -------
    pd.DataFrame with columns ['ds', 'y', optional regressors]
    """
    if product_name is not None:
        df = weekly_df[weekly_df["product_name"] == product_name].copy()
    else:
        df = weekly_df.groupby("week_start")["demand"].sum().reset_index()

    df = df.rename(columns={"week_start": "ds", "demand": "y"})

    # Add regressors if they exist
    regressor_candidates = [
        "is_holiday_season", "is_q4", "is_year_end",
        "is_summer", "quarter",
    ]
    available_regressors = [c for c in regressor_candidates if c in df.columns]
    cols_to_keep = ["ds", "y"] + available_regressors
    df = df[[c for c in cols_to_keep if c in df.columns]].sort_values("ds")

    logger.info("Prophet DataFrame built: %d rows.", len(df))
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from preprocessing.preprocess().

    Returns
    -------
    (feature_df, prophet_df) tuple
        feature_df : wide feature matrix for XGBoost/sklearn
        prophet_df : ds/y DataFrame aggregated across all products for Prophet
    """
    weekly = aggregate_weekly(df)
    weekly = add_lag_features(weekly)
    weekly = add_rolling_features(weekly)
    weekly = add_seasonality_features(weekly)
    weekly = add_lead_time_features(weekly)
    weekly = encode_group_columns(weekly)

    prophet_df = build_prophet_df(weekly)

    # Drop rows where lag features are NaN (the first LAG_PERIODS weeks per product)
    lag_cols = [c for c in weekly.columns if "_lag_" in c]
    weekly_clean = weekly.dropna(subset=lag_cols).reset_index(drop=True)

    logger.info("Feature matrix: %d rows, %d features", *weekly_clean.shape)
    return weekly_clean, prophet_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.data_loader import load_raw
    from src.preprocessing import preprocess

    raw = load_raw()
    clean = preprocess(raw)
    feat_df, prop_df = build_feature_matrix(clean)
    print("Feature matrix shape:", feat_df.shape)
    print("Prophet df shape:", prop_df.shape)
    print(feat_df.columns.tolist())
