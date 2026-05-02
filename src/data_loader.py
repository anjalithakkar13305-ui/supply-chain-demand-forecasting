"""
data_loader.py
--------------
Loads the raw supply chain CSV into memory and pushes it to SQLite.
Handles encoding quirks in the DataCo dataset and validates the schema
before any downstream processing touches the data.
"""

import os
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "supply_chain.csv"
DB_PATH = ROOT / "data" / "supply_chain.db"

# Columns that must exist for the pipeline to work
REQUIRED_COLUMNS = {
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Order Id",
    "Order Item Quantity",
    "Sales",
    "Product Name",
    "Category Name",
    "Order Region",
    "Market",
    "Shipping Mode",
    "Order Status",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Order Item Product Price",
    "Order Profit Per Order",
}


def load_raw(path: str | Path = DATA_PATH,
             chunksize: int = 20000) -> pd.DataFrame:
    """
    Read the raw CSV in chunks to stay within available RAM,
    then concatenate into a single DataFrame.

    Parameters
    ----------
    path : path-like
        Path to the supply_chain.csv file.
    chunksize : int
        Rows per chunk. Lower this (e.g. 10000) if you still get OOM.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at *path*.
    ValueError
        If required columns are missing after load.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. "
                                "Place supply_chain.csv in the data/ folder.")

    logger.info("Loading dataset from %s (chunked, %d rows/chunk) …",
                path, chunksize)

    chunks = []
    total  = 0
    for chunk in pd.read_csv(path, encoding="latin1", low_memory=True,
                              chunksize=chunksize):
        # Downcast numerics immediately to cut memory usage
        for col in chunk.select_dtypes(include="float64").columns:
            chunk[col] = chunk[col].astype("float32")
        for col in chunk.select_dtypes(include="int64").columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast="integer")
        chunks.append(chunk)
        total += len(chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    # ── Schema validation ──────────────────────────────────────────────────────
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    logger.info("Schema validation passed.")
    return df


def push_to_sqlite(df: pd.DataFrame, db_path: str | Path = DB_PATH,
                   table: str = "orders", if_exists: str = "replace") -> None:
    """
    Write *df* to a SQLite database table.

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    db_path : path-like
        Path to the SQLite database file (created if absent).
    table : str
        Target table name.
    if_exists : {'replace', 'append', 'fail'}
        Behaviour if the table already exists.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing %d rows to SQLite → %s [table=%s]", len(df), db_path, table)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists=if_exists, index=False, chunksize=5000)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_order_date ON {table} "
                     f"([order date (DateOrders)])")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_product ON {table} "
                     f"([Product Name])")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_region ON {table} "
                     f"([Order Region])")
        conn.commit()
        logger.info("SQLite write complete. Indexes created.")
    finally:
        conn.close()


def load_from_sqlite(db_path: str | Path = DB_PATH,
                     table: str = "orders") -> pd.DataFrame:
    """
    Load the full orders table from SQLite back into a DataFrame.

    Parameters
    ----------
    db_path : path-like
    table : str

    Returns
    -------
    pd.DataFrame
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. "
                                "Run push_to_sqlite first.")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()
    logger.info("Loaded %d rows from SQLite.", len(df))
    return df


def run_pipeline(csv_path: str | Path = DATA_PATH,
                 db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """
    End-to-end: load CSV → validate → push to SQLite → return DataFrame.

    This is the main entry point used by downstream modules.
    """
    df = load_raw(csv_path)
    push_to_sqlite(df, db_path)
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(df.shape)
    print(df.dtypes)
