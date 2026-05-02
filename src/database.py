"""
database.py
-----------
SQL query layer on top of the SQLite database populated by data_loader.py.
All business-level aggregations live here so downstream code never writes
raw SQL outside this module.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "supply_chain.db"


# ── Connection helper ──────────────────────────────────────────────────────────

def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run data_loader.run_pipeline() first."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple = (), db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Execute an arbitrary SELECT query and return results as a DataFrame.

    Parameters
    ----------
    sql : str
        SQL query string.
    params : tuple
        Positional parameters for parameterised queries.
    db_path : Path

    Returns
    -------
    pd.DataFrame
    """
    conn = _get_conn(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


# ── Pre-built business queries ─────────────────────────────────────────────────

def total_demand_by_region(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Total ordered quantity and revenue grouped by Order Region.

    Returns
    -------
    pd.DataFrame
        Columns: order_region, total_quantity, total_sales, order_count
    """
    sql = """
        SELECT
            [Order Region]              AS order_region,
            SUM([Order Item Quantity])  AS total_quantity,
            ROUND(SUM([Sales]), 2)      AS total_sales,
            COUNT([Order Id])           AS order_count
        FROM orders
        GROUP BY [Order Region]
        ORDER BY total_quantity DESC
    """
    df = query(sql, db_path=db_path)
    logger.info("total_demand_by_region → %d regions", len(df))
    return df


def demand_by_category(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Total demand aggregated by product category.

    Returns
    -------
    pd.DataFrame
        Columns: category_name, total_quantity, total_sales, avg_profit_ratio
    """
    sql = """
        SELECT
            [Category Name]                         AS category_name,
            SUM([Order Item Quantity])              AS total_quantity,
            ROUND(SUM([Sales]), 2)                  AS total_sales,
            ROUND(AVG([Order Item Profit Ratio]), 4) AS avg_profit_ratio
        FROM orders
        GROUP BY [Category Name]
        ORDER BY total_quantity DESC
    """
    df = query(sql, db_path=db_path)
    logger.info("demand_by_category → %d categories", len(df))
    return df


def monthly_demand_trend(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Monthly order quantity and revenue trend.

    Parses the order date column (MM/DD/YYYY HH:MM) via SQLite's
    strftime so the aggregation stays inside the database.

    Returns
    -------
    pd.DataFrame
        Columns: year_month, total_quantity, total_sales, order_count
    """
    sql = """
        SELECT
            strftime('%Y-%m',
                substr([order date (DateOrders)], 7, 4) || '-' ||
                substr([order date (DateOrders)], 1, 2) || '-' ||
                substr([order date (DateOrders)], 4, 2)
            )                               AS year_month,
            SUM([Order Item Quantity])      AS total_quantity,
            ROUND(SUM([Sales]), 2)          AS total_sales,
            COUNT([Order Id])               AS order_count
        FROM orders
        WHERE [order date (DateOrders)] IS NOT NULL
        GROUP BY year_month
        ORDER BY year_month
    """
    df = query(sql, db_path=db_path)
    logger.info("monthly_demand_trend → %d months", len(df))
    return df


def top_products_by_demand(top_n: int = 20, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Top N products by total ordered quantity.

    Parameters
    ----------
    top_n : int
        How many products to return.

    Returns
    -------
    pd.DataFrame
        Columns: product_name, category_name, total_quantity, total_sales
    """
    sql = """
        SELECT
            [Product Name]              AS product_name,
            [Category Name]             AS category_name,
            SUM([Order Item Quantity])  AS total_quantity,
            ROUND(SUM([Sales]), 2)      AS total_sales
        FROM orders
        GROUP BY [Product Name], [Category Name]
        ORDER BY total_quantity DESC
        LIMIT ?
    """
    df = query(sql, params=(top_n,), db_path=db_path)
    logger.info("top_products_by_demand (top=%d) → %d rows", top_n, len(df))
    return df


def late_delivery_summary(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Late-delivery rate by shipping mode and region — useful for
    supply-chain risk analysis.

    Returns
    -------
    pd.DataFrame
        Columns: shipping_mode, order_region, total_orders,
                 late_orders, late_rate_pct
    """
    sql = """
        SELECT
            [Shipping Mode]                                 AS shipping_mode,
            [Order Region]                                  AS order_region,
            COUNT(*)                                        AS total_orders,
            SUM([Late_delivery_risk])                       AS late_orders,
            ROUND(100.0 * SUM([Late_delivery_risk])
                  / COUNT(*), 2)                            AS late_rate_pct
        FROM orders
        GROUP BY [Shipping Mode], [Order Region]
        ORDER BY late_rate_pct DESC
    """
    df = query(sql, db_path=db_path)
    logger.info("late_delivery_summary → %d rows", len(df))
    return df


def demand_by_market_and_segment(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Demand breakdown by market and customer segment.

    Returns
    -------
    pd.DataFrame
    """
    sql = """
        SELECT
            [Market]                        AS market,
            [Customer Segment]              AS customer_segment,
            SUM([Order Item Quantity])      AS total_quantity,
            ROUND(SUM([Sales]), 2)          AS total_sales,
            ROUND(AVG([Sales per customer]), 2) AS avg_sales_per_customer
        FROM orders
        GROUP BY [Market], [Customer Segment]
        ORDER BY total_quantity DESC
    """
    return query(sql, db_path=db_path)


def inventory_risk_query(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Products where scheduled shipping days consistently fall short of
    actual shipping days — a proxy for inventory / fulfilment risk.

    Returns
    -------
    pd.DataFrame
        Columns: product_name, category_name, avg_scheduled_days,
                 avg_actual_days, avg_delay, order_count
    """
    sql = """
        SELECT
            [Product Name]                                      AS product_name,
            [Category Name]                                     AS category_name,
            ROUND(AVG([Days for shipment (scheduled)]), 2)      AS avg_scheduled_days,
            ROUND(AVG([Days for shipping (real)]), 2)           AS avg_actual_days,
            ROUND(AVG([Days for shipping (real)] -
                      [Days for shipment (scheduled)]), 2)      AS avg_delay,
            COUNT(*)                                            AS order_count
        FROM orders
        GROUP BY [Product Name], [Category Name]
        HAVING order_count >= 20
        ORDER BY avg_delay DESC
        LIMIT 50
    """
    return query(sql, db_path=db_path)


if __name__ == "__main__":
    print("=== Region Demand ===")
    print(total_demand_by_region().head())

    print("\n=== Category Demand ===")
    print(demand_by_category().head())

    print("\n=== Monthly Trend ===")
    print(monthly_demand_trend().head(10))

    print("\n=== Top Products ===")
    print(top_products_by_demand(5))

    print("\n=== Inventory Risk ===")
    print(inventory_risk_query().head())
