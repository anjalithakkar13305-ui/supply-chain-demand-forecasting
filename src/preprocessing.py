"""
preprocessing.py
----------------
Cleans, validates, and type-converts the raw supply chain DataFrame.

Steps performed
---------------
1. Rename columns to clean snake_case names.
2. Parse date columns (MM/DD/YYYY HH:MM format used by DataCo).
3. Drop columns that are either all-null or carry no signal for forecasting
   (PII like email/password, image URLs, redundant IDs).
4. Handle missing values with context-aware strategies.
5. Clip / cap outliers using IQR-based bounds on numeric columns.
6. Encode low-cardinality categoricals as category dtype.
7. Add a binary `is_late` indicator from delivery status.

The output DataFrame is used directly by feature_engineering.py.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Column renames: original → snake_case ─────────────────────────────────────
RENAME_MAP = {
    "Type": "payment_type",
    "Days for shipping (real)": "days_shipping_real",
    "Days for shipment (scheduled)": "days_shipping_scheduled",
    "Benefit per order": "benefit_per_order",
    "Sales per customer": "sales_per_customer",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
    "Category Id": "category_id",
    "Category Name": "category_name",
    "Customer City": "customer_city",
    "Customer Country": "customer_country",
    "Customer Id": "customer_id",
    "Customer Segment": "customer_segment",
    "Customer State": "customer_state",
    "Customer Zipcode": "customer_zipcode",
    "Department Id": "department_id",
    "Department Name": "department_name",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Market": "market",
    "Order City": "order_city",
    "Order Country": "order_country",
    "Order Customer Id": "order_customer_id",
    "order date (DateOrders)": "order_date",
    "Order Id": "order_id",
    "Order Item Cardprod Id": "product_card_id",
    "Order Item Discount": "item_discount",
    "Order Item Discount Rate": "item_discount_rate",
    "Order Item Id": "order_item_id",
    "Order Item Product Price": "item_product_price",
    "Order Item Profit Ratio": "item_profit_ratio",
    "Order Item Quantity": "order_quantity",
    "Sales": "sales",
    "Order Item Total": "order_item_total",
    "Order Profit Per Order": "order_profit",
    "Order Region": "order_region",
    "Order State": "order_state",
    "Order Status": "order_status",
    "Order Zipcode": "order_zipcode",
    "Product Card Id": "product_card_id_2",
    "Product Category Id": "product_category_id",
    "Product Name": "product_name",
    "Product Price": "product_price",
    "Product Status": "product_status",
    "shipping date (DateOrders)": "shipping_date",
    "Shipping Mode": "shipping_mode",
}

# Columns to drop — PII, redundant IDs, URL blobs, constant or near-useless
DROP_COLS = [
    "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
    "Customer Street", "Product Image", "Product Description",
    "Order Zipcode", "Customer Zipcode",
    "Order Customer Id",        # duplicate of customer_id
    "product_card_id_2",        # same as product_card_id after rename
]

# Numeric columns where outlier capping makes sense
NUMERIC_OUTLIER_COLS = [
    "sales", "order_profit", "benefit_per_order",
    "order_item_total", "item_product_price", "product_price",
    "item_discount", "sales_per_customer",
]

# Low-cardinality categoricals to encode
CATEGORICAL_COLS = [
    "payment_type", "delivery_status", "category_name", "customer_segment",
    "department_name", "market", "shipping_mode", "order_status",
    "order_region",
]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw column names to clean snake_case equivalents."""
    df = df.rename(columns=RENAME_MAP)
    logger.info("Columns renamed.")
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop PII, redundant, and structurally useless columns."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    # Also drop any all-null columns
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        logger.warning("Dropping all-null columns: %s", all_null)
        df = df.drop(columns=all_null)
    logger.info("Dropped %d columns.", len(cols_to_drop) + len(all_null))
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse order_date and shipping_date from MM/DD/YYYY HH:MM format.
    Adds derived columns: order_year, order_month, order_week, order_dow.
    """
    for col in ["order_date", "shipping_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %H:%M",
                                      errors="coerce")
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.warning("%s: %d unparseable dates (set to NaT).",
                               col, null_count)

    if "order_date" in df.columns:
        df["order_year"]  = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month
        df["order_week"]  = df["order_date"].dt.isocalendar().week.astype("Int64")
        df["order_dow"]   = df["order_date"].dt.dayofweek   # 0=Monday

    if "order_date" in df.columns and "shipping_date" in df.columns:
        df["actual_lead_time"] = (
            df["shipping_date"] - df["order_date"]
        ).dt.days

    logger.info("Dates parsed. order_date range: %s → %s",
                df["order_date"].min(), df["order_date"].max())
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill or drop missing values with context-appropriate strategies.

    - Numeric columns: fill with column median (robust to skew).
    - Categorical columns: fill with 'Unknown'.
    - Rows where order_date is NaT are dropped (they are useless for time-series).
    """
    before = len(df)
    if "order_date" in df.columns:
        df = df.dropna(subset=["order_date"])
        logger.info("Dropped %d rows with missing order_date.", before - len(df))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_cnt = df[col].isna().sum()
        if null_cnt > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Filled %d NaNs in '%s' with median %.4f",
                         null_cnt, col, median_val)

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        null_cnt = df[col].isna().sum()
        if null_cnt > 0:
            df[col] = df[col].fillna("Unknown")

    logger.info("Missing value imputation complete. Remaining nulls: %d",
                df.isna().sum().sum())
    return df


def cap_outliers(df: pd.DataFrame,
                 cols: list[str] = NUMERIC_OUTLIER_COLS,
                 iqr_factor: float = 3.0) -> pd.DataFrame:
    """
    Cap extreme outliers using IQR-based bounds (Tukey fences).

    Uses iqr_factor=3.0 (more permissive than the usual 1.5) because
    supply chain data legitimately has high-value bulk orders.

    Parameters
    ----------
    cols : list[str]
        Numeric columns to cap.
    iqr_factor : float
        Number of IQRs beyond Q1/Q3 to use as bounds.
    """
    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if clipped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.debug("Capped %d outliers in '%s' [%.2f, %.2f]",
                         clipped, col, lower, upper)
    logger.info("Outlier capping complete.")
    return df


def encode_categoricals(df: pd.DataFrame,
                         cols: list[str] = CATEGORICAL_COLS) -> pd.DataFrame:
    """
    Cast low-cardinality string columns to pandas category dtype.
    Also adds numeric label-encoded versions (suffix _enc) for tree models.
    """
    le = LabelEncoder()
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    logger.info("Categorical encoding complete.")
    return df


def add_derived_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary indicator columns derived from existing data.

    - is_late: 1 if delivery_status contains 'Late', else 0
    - is_cancelled: 1 if order_status == 'CANCELED'
    - is_discounted: 1 if item_discount_rate > 0
    """
    if "delivery_status" in df.columns:
        df["is_late"] = df["delivery_status"].astype(str).str.contains(
            "Late", case=False, na=False
        ).astype(int)

    if "order_status" in df.columns:
        df["is_cancelled"] = (df["order_status"] == "CANCELED").astype(int)

    if "item_discount_rate" in df.columns:
        df["is_discounted"] = (df["item_discount_rate"] > 0).astype(int)

    logger.info("Derived flags added.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline — apply all steps in order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from data_loader.load_raw().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df = rename_columns(df)
    df = drop_useless_columns(df)
    df = parse_dates(df)
    df = handle_missing_values(df)
    df = cap_outliers(df)
    df = encode_categoricals(df)
    df = add_derived_flags(df)

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.data_loader import load_raw

    raw = load_raw()
    clean = preprocess(raw)
    print(clean.shape)
    print(clean.dtypes)
    print(clean.head(3))
