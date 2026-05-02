"""
train_model.py
--------------
Trains and compares two forecasting models:
  1. XGBoost Regressor  — tree-based, handles non-linearity, lag features
  2. Prophet            — additive time-series model, handles seasonality

Workflow
--------
1. Build feature matrix from preprocessed data.
2. Time-series train/test split (last 12 weeks as test set).
3. Train both models.
4. Evaluate on hold-out set via evaluate_model.compute_metrics().
5. Save both models; mark the winner as "best_model".

Saved artefacts
---------------
models/xgboost_model.json
models/prophet_model.pkl
models/best_model.pkl       ← whichever wins on RMSE
models/model_metadata.json  ← metrics + selected model
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Feature columns for XGBoost ───────────────────────────────────────────────
FEATURE_COLS_BASE = [
    "month_sin", "month_cos", "week_of_year_sin", "week_of_year_cos",
    "quarter", "is_q4", "is_year_end", "is_summer", "is_holiday_season",
    "year", "week_of_year",
    "product_name_enc", "category_name_enc", "order_region_enc", "market_enc",
    "days_shipping_real", "days_shipping_scheduled",
    "shipping_delay", "delay_ratio",
    "is_late", "is_chronically_late",
    "item_discount_rate", "order_count",
]

LAG_COLS_PATTERN  = "_lag_"
ROLL_COLS_PATTERN = "_roll_"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the intersection of desired features and available DataFrame columns."""
    lag_cols  = [c for c in df.columns if LAG_COLS_PATTERN in c]
    roll_cols = [c for c in df.columns if ROLL_COLS_PATTERN in c]
    base = [c for c in FEATURE_COLS_BASE if c in df.columns]
    all_features = list(dict.fromkeys(base + lag_cols + roll_cols))
    return all_features


def time_split(df: pd.DataFrame,
               test_weeks: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split weekly DataFrame into train/test by time.
    The last *test_weeks* unique weeks form the test set.

    Parameters
    ----------
    df : pd.DataFrame
    test_weeks : int

    Returns
    -------
    (train, test) DataFrames
    """
    all_weeks = sorted(df["week_start"].unique())
    cutoff = all_weeks[-test_weeks]
    train = df[df["week_start"] < cutoff].copy()
    test  = df[df["week_start"] >= cutoff].copy()
    logger.info("Train: %d rows (%s → %s) | Test: %d rows (%s → %s)",
                len(train), train["week_start"].min(), train["week_start"].max(),
                len(test),  test["week_start"].min(),  test["week_start"].max())
    return train, test


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(train: pd.DataFrame,
                  feature_cols: list[str],
                  target_col: str = "demand"):
    """
    Train an XGBoost regressor with a light hyperparameter set tuned for
    weekly supply-chain demand.

    Parameters
    ----------
    train : pd.DataFrame
    feature_cols : list[str]
    target_col : str

    Returns
    -------
    xgb.XGBRegressor
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    X_train = train[feature_cols].fillna(0)
    y_train = train[target_col].clip(lower=0)

    # Use last 10% of training set as validation for early stopping
    val_size     = max(1, int(0.1 * len(X_train)))
    X_tr, X_val  = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val  = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    # ── Hyperparameter search ──────────────────────────────────────────────────
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer, mean_squared_error

    param_dist = {
        "n_estimators":     [300, 500, 700],
        "learning_rate":    [0.01, 0.05, 0.1],
        "max_depth":        [4, 6, 8],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [3, 5, 10],
        "reg_alpha":        [0.0, 0.1, 0.5],
        "reg_lambda":       [0.5, 1.0, 2.0],
    }

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )

    rmse_scorer = make_scorer(
        mean_squared_error, greater_is_better=False, squared=False
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,               # 20 random combinations
        scoring=rmse_scorer,
        cv=3,                    # 3-fold time-series aware CV
        random_state=42,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    logger.info("Running hyperparameter search (20 iterations × 3-fold CV) …")
    search.fit(X_tr, y_tr)
    best_params = search.best_params_
    logger.info("Best params: %s", best_params)
    logger.info("Best CV RMSE: %.4f", -search.best_score_)

    # ── Refit best model with early stopping on validation set ────────────────
    model = xgb.XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("XGBoost trained. Best iteration: %d", model.best_iteration)
    return model


def predict_xgboost(model, test: pd.DataFrame, feature_cols: list[str],
                    target_col: str = "demand") -> pd.DataFrame:
    """Generate predictions on the test set and return a results DataFrame."""
    X_test = test[feature_cols].fillna(0)
    preds  = model.predict(X_test).clip(0)

    results = test[["week_start", "product_name", target_col]].copy()
    results["predicted"] = preds
    return results


def save_xgboost(model, path: Path = MODELS_DIR / "xgboost_model.json"):
    """Save XGBoost model to JSON format."""
    model.save_model(str(path))
    logger.info("XGBoost model saved → %s", path)


def load_xgboost(path: Path = MODELS_DIR / "xgboost_model.json"):
    """Load a previously saved XGBoost model."""
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


# ── Prophet ───────────────────────────────────────────────────────────────────

def train_prophet(prophet_df: pd.DataFrame,
                  regressors: list[str] | None = None):
    """
    Train a Prophet model on total weekly demand.

    Parameters
    ----------
    prophet_df : pd.DataFrame
        Must have columns ['ds', 'y'] plus optional regressor columns.
    regressors : list[str] or None
        Extra regressor column names to include.

    Returns
    -------
    prophet.forecaster.Prophet
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("prophet is not installed. Run: pip install prophet")

    if regressors is None:
        regressors = [c for c in ["is_holiday_season", "is_q4", "is_year_end",
                                   "is_summer"]
                      if c in prophet_df.columns]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # data is already weekly
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.9,
    )
    for reg in regressors:
        model.add_regressor(reg)

    # Prophet wants no NaNs in regressors
    fit_df = prophet_df.fillna(0)
    model.fit(fit_df)
    logger.info("Prophet model trained on %d weeks.", len(fit_df))
    return model


def predict_prophet(model, future_periods: int = 12,
                    prophet_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Generate Prophet forecasts for *future_periods* weeks.

    Parameters
    ----------
    model : Prophet
    future_periods : int
        Weeks to forecast beyond the training data.
    prophet_df : pd.DataFrame or None
        If given, used to create the full future DataFrame with regressors.

    Returns
    -------
    pd.DataFrame  (Prophet forecast output)
    """
    future = model.make_future_dataframe(periods=future_periods, freq="W")

    # Fill regressor columns with their last known value (forward fill proxy)
    if prophet_df is not None:
        reg_cols = [c for c in prophet_df.columns if c not in ("ds", "y")]
        for col in reg_cols:
            if col in future.columns:
                continue
            last_val = prophet_df[col].iloc[-1] if len(prophet_df) > 0 else 0
            future[col] = last_val

    forecast = model.predict(future.fillna(0))
    return forecast


def save_prophet(model, path: Path = MODELS_DIR / "prophet_model.pkl"):
    """Pickle the Prophet model."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Prophet model saved → %s", path)


def load_prophet(path: Path = MODELS_DIR / "prophet_model.pkl"):
    """Load a pickled Prophet model."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Comparison & selection ────────────────────────────────────────────────────

def compare_models(xgb_results: pd.DataFrame,
                   prophet_forecast: pd.DataFrame,
                   test: pd.DataFrame,
                   prophet_df_full: pd.DataFrame) -> dict:
    """
    Compare XGBoost and Prophet on the hold-out test set.

    Returns a dict with metrics for both models.
    """
    from src.evaluate_model import compute_metrics

    # XGBoost metrics
    xgb_metrics = compute_metrics(
        xgb_results["demand"].values,
        xgb_results["predicted"].values,
        label="XGBoost",
    )

    # Prophet: align forecast to test period dates
    test_dates = test["week_start"].unique()
    prophet_test = prophet_forecast[
        prophet_forecast["ds"].isin(pd.to_datetime(test_dates))
    ][["ds", "yhat"]].copy()

    # Aggregate actual demand to match Prophet's total-demand scope
    actual_agg = (
        test.groupby("week_start")["demand"].sum().reset_index()
    )
    actual_agg["week_start"] = pd.to_datetime(actual_agg["week_start"])
    merged = actual_agg.merge(
        prophet_test, left_on="week_start", right_on="ds", how="inner"
    )

    if len(merged) > 0:
        prophet_metrics = compute_metrics(
            merged["demand"].values,
            merged["yhat"].clip(0).values,
            label="Prophet",
        )
    else:
        logger.warning("Could not align Prophet forecast to test dates.")
        prophet_metrics = {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf}

    comparison = {
        "xgboost": xgb_metrics,
        "prophet": prophet_metrics,
    }
    logger.info("Model comparison → XGBoost RMSE: %.2f | Prophet RMSE: %.2f",
                xgb_metrics["RMSE"], prophet_metrics["RMSE"])
    return comparison


# ── Full training pipeline ────────────────────────────────────────────────────

def run_training_pipeline():
    """
    End-to-end training pipeline.

    1. Load → preprocess → build features.
    2. Train XGBoost + Prophet.
    3. Evaluate and compare.
    4. Save both models and write metadata.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data_loader import load_raw
    from src.preprocessing import preprocess
    from src.feature_engineering import build_feature_matrix
    from src.evaluate_model import compute_metrics, plot_forecast_vs_actual

    logger.info("=== Starting Training Pipeline ===")

    # Step 1: Data
    raw = load_raw()
    clean = preprocess(raw)
    feature_df, prophet_df = build_feature_matrix(clean)

    # Step 2: Split
    feature_cols = get_feature_cols(feature_df)
    train, test = time_split(feature_df)

    # Step 3: XGBoost
    logger.info("Training XGBoost …")
    xgb_model = train_xgboost(train, feature_cols)
    xgb_results = predict_xgboost(xgb_model, test, feature_cols)
    save_xgboost(xgb_model)

    # Step 4: Prophet (on total weekly demand)
    logger.info("Training Prophet …")
    prophet_model = train_prophet(prophet_df)

    test_weeks = len(test["week_start"].unique())
    prophet_forecast = predict_prophet(prophet_model, test_weeks, prophet_df)
    save_prophet(prophet_model)

    # Step 5: Compare
    comparison = compare_models(xgb_results, prophet_forecast, test, prophet_df)

    # Step 6: Save best model
    if comparison["xgboost"]["RMSE"] <= comparison["prophet"]["RMSE"]:
        best_name = "xgboost"
        best_model = xgb_model
        logger.info("Winner: XGBoost")
    else:
        best_name = "prophet"
        best_model = prophet_model
        logger.info("Winner: Prophet")

    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump({"name": best_name, "model": best_model,
                     "feature_cols": feature_cols}, f)

    # Step 7: Save metadata
    metadata = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "metrics": comparison,
        "train_size": len(train),
        "test_size": len(test),
        "n_products": feature_df["product_name"].nunique(),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → models/model_metadata.json")

    # Step 8: Forecast vs Actual plot
    plot_forecast_vs_actual(
        xgb_results["demand"].values,
        xgb_results["predicted"].values,
        label="XGBoost — All Products (Test Set)",
    )

    logger.info("=== Training Pipeline Complete ===")
    return metadata


if __name__ == "__main__":
    results = run_training_pipeline()
    print(json.dumps(results, indent=2))
