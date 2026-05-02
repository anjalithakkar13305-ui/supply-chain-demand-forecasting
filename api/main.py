"""
api/main.py
-----------
FastAPI application serving demand forecasts.

Endpoints
---------
GET  /health                  — liveness check
POST /forecast/product        — single-product 30/60/90-day forecast
POST /forecast/bulk           — bulk forecast for all products in a category

Usage
-----
    uvicorn api.main:app --reload --port 8000
"""

import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PATH  = ROOT / "data" / "supply_chain.db"

app = FastAPI(
    title="Supply Chain Demand Forecasting API",
    description="REST API for weekly demand forecasts powered by XGBoost and Prophet.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model loading (cached at startup) ─────────────────────────────────────────

class ModelRegistry:
    """Lazy-loaded model registry — models are loaded once on first request."""

    _best_bundle: dict | None = None
    _metadata: dict | None = None
    _feature_df: pd.DataFrame | None = None
    _prophet_df: pd.DataFrame | None = None

    @classmethod
    def get_best_bundle(cls) -> dict:
        if cls._best_bundle is None:
            path = MODELS_DIR / "best_model.pkl"
            if not path.exists():
                raise RuntimeError(
                    "No trained model found. Run src/train_model.py first."
                )
            with open(path, "rb") as f:
                cls._best_bundle = pickle.load(f)
            logger.info("Loaded best model: %s", cls._best_bundle["name"])
        return cls._best_bundle

    @classmethod
    def get_metadata(cls) -> dict:
        if cls._metadata is None:
            path = MODELS_DIR / "model_metadata.json"
            if path.exists():
                with open(path) as f:
                    cls._metadata = json.load(f)
            else:
                cls._metadata = {}
        return cls._metadata

    @classmethod
    def get_feature_data(cls) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and cache the full feature matrix (used for inference context)."""
        if cls._feature_df is None:
            import sys
            sys.path.insert(0, str(ROOT))
            from src.data_loader import load_raw
            from src.preprocessing import preprocess
            from src.feature_engineering import build_feature_matrix
            logger.info("Building feature matrix for inference …")
            raw = load_raw()
            clean = preprocess(raw)
            cls._feature_df, cls._prophet_df = build_feature_matrix(clean)
        return cls._feature_df, cls._prophet_df


# ── Request / Response schemas ────────────────────────────────────────────────

class ProductForecastRequest(BaseModel):
    product_name: str = Field(..., example="Smart watch",
                               description="Exact product name as in the dataset.")
    horizon_days: int = Field(30, ge=7, le=180,
                               description="Forecast horizon in days (7–180).")
    region: Optional[str] = Field(None, example="Southeast Asia",
                                   description="Optional filter by order region.")


class ForecastPoint(BaseModel):
    date: str
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class ProductForecastResponse(BaseModel):
    product_name: str
    model_used: str
    horizon_days: int
    forecast: list[ForecastPoint]
    historical_avg_weekly_demand: float
    generated_at: str


class BulkForecastRequest(BaseModel):
    category_name: str = Field(..., example="Sporting Goods",
                                description="Product category to forecast.")
    horizon_days: int = Field(30, ge=7, le=90)


class BulkForecastItem(BaseModel):
    product_name: str
    total_predicted_demand: float
    forecast_weeks: int


class BulkForecastResponse(BaseModel):
    category_name: str
    horizon_days: int
    products: list[BulkForecastItem]
    generated_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _horizon_weeks(horizon_days: int) -> int:
    return max(1, round(horizon_days / 7))


def _xgb_product_forecast(model, feature_df: pd.DataFrame,
                            feature_cols: list[str],
                            product_name: str,
                            horizon_weeks: int,
                            region: str | None) -> list[dict]:
    """
    Iterative XGBoost forecast for a single product.

    Uses the last known feature row and rolls it forward week-by-week,
    updating lag features at each step.
    """
    mask = feature_df["product_name"] == product_name
    if region:
        mask &= feature_df["order_region"] == region

    prod_df = feature_df[mask].sort_values("week_start")
    if len(prod_df) == 0:
        raise HTTPException(status_code=404,
                            detail=f"Product '{product_name}' not found.")

    last_row   = prod_df.iloc[-1].copy()
    last_week  = pd.to_datetime(last_row["week_start"])
    lag_cols   = sorted([c for c in feature_cols if "_lag_" in c],
                        key=lambda c: int(c.split("_lag_")[1].replace("w", "")))
    roll_cols  = [c for c in feature_cols if "_roll_" in c]

    history = list(prod_df["demand"].values[-16:])   # keep last 16 weeks
    results = []

    for step in range(1, horizon_weeks + 1):
        next_week = last_week + timedelta(weeks=step)
        row = last_row.copy()

        # Update date features
        row["week_of_year"]     = next_week.isocalendar()[1]
        row["year"]             = next_week.year
        month = next_week.month
        woy   = float(row["week_of_year"])
        row["month_sin"]        = np.sin(2 * np.pi * month / 12)
        row["month_cos"]        = np.cos(2 * np.pi * month / 12)
        row["week_of_year_sin"] = np.sin(2 * np.pi * woy / 52)
        row["week_of_year_cos"] = np.cos(2 * np.pi * woy / 52)
        row["quarter"]          = (month - 1) // 3 + 1
        row["is_q4"]            = int(month >= 10)
        row["is_year_end"]      = int(month >= 11)
        row["is_summer"]        = int(month in [6, 7, 8])
        row["is_holiday_season"]= int(woy >= 47)

        # Update lag features from history
        for lc in lag_cols:
            lag_n = int(lc.split("_lag_")[1].replace("w", ""))
            idx   = -(lag_n)
            row[lc] = history[idx] if abs(idx) <= len(history) else 0.0

        # Update rolling features
        for rc in roll_cols:
            parts = rc.split("_roll_")
            window = int(parts[1].replace("w", "").split("_")[0])
            window_history = history[-window:] if len(history) >= window else history
            if "mean" in rc:
                row[rc] = float(np.mean(window_history))
            elif "std" in rc:
                row[rc] = float(np.std(window_history)) if len(window_history) > 1 else 0.0

        X = pd.DataFrame([row[feature_cols].fillna(0).values],
                         columns=feature_cols)
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)

        # Confidence interval: ±1.5 * rolling std (simple heuristic)
        roll_std = float(np.std(history[-4:])) if len(history) >= 2 else pred * 0.15
        results.append({
            "date": next_week.strftime("%Y-%m-%d"),
            "predicted_demand": round(pred, 2),
            "lower_bound": round(max(0.0, pred - 1.5 * roll_std), 2),
            "upper_bound": round(pred + 1.5 * roll_std, 2),
        })
        history.append(pred)

    return results


def _prophet_product_forecast(model, prophet_df: pd.DataFrame,
                               horizon_weeks: int) -> list[dict]:
    """Generate a Prophet forecast for total demand."""
    future = model.make_future_dataframe(periods=horizon_weeks, freq="W")
    reg_cols = [c for c in prophet_df.columns if c not in ("ds", "y")]
    for col in reg_cols:
        if col not in future.columns:
            future[col] = prophet_df[col].iloc[-1] if len(prophet_df) > 0 else 0
    forecast = model.predict(future.fillna(0))
    future_part = forecast.tail(horizon_weeks)
    results = []
    for _, row in future_part.iterrows():
        results.append({
            "date": pd.Timestamp(row["ds"]).strftime("%Y-%m-%d"),
            "predicted_demand": round(max(0.0, row["yhat"]), 2),
            "lower_bound": round(max(0.0, row["yhat_lower"]), 2),
            "upper_bound": round(max(0.0, row["yhat_upper"]), 2),
        })
    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"], include_in_schema=False)
def root():
    """Redirect root to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
def health_check():
    """
    Liveness check.

    Returns model status, best model name, and current timestamp.
    """
    meta = ModelRegistry.get_metadata()
    model_ready = (MODELS_DIR / "best_model.pkl").exists()
    return {
        "status": "ok",
        "model_ready": model_ready,
        "best_model": meta.get("best_model", "unknown"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/forecast/product", response_model=ProductForecastResponse,
          tags=["Forecasting"])
def forecast_single_product(req: ProductForecastRequest):
    """
    Forecast demand for a single product over the requested horizon.

    - Uses XGBoost for product-level iterative forecasting.
    - Falls back to Prophet total-demand scaling if the product is not found
      in the XGBoost feature matrix.
    - Horizon accepts 7–180 days; internally rounds to full weeks.

    **Example request body:**
    ```json
    {
      "product_name": "Smart watch",
      "horizon_days": 30,
      "region": "Southeast Asia"
    }
    ```
    """
    bundle = ModelRegistry.get_best_bundle()
    feature_df, prophet_df = ModelRegistry.get_feature_data()
    horizon_weeks = _horizon_weeks(req.horizon_days)

    model_name = bundle["name"]

    if model_name == "xgboost":
        import xgboost as xgb   # noqa: F401
        model        = bundle["model"]
        feature_cols = bundle["feature_cols"]
        try:
            forecast_points = _xgb_product_forecast(
                model, feature_df, feature_cols,
                req.product_name, horizon_weeks, req.region,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("XGBoost inference error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    else:
        model           = bundle["model"]
        forecast_points = _prophet_product_forecast(model, prophet_df, horizon_weeks)

    # Historical weekly average for context
    mask = feature_df["product_name"] == req.product_name
    if req.region:
        mask &= feature_df["order_region"] == req.region
    hist_avg = float(feature_df[mask]["demand"].mean()) if mask.any() else 0.0

    return ProductForecastResponse(
        product_name=req.product_name,
        model_used=model_name,
        horizon_days=req.horizon_days,
        forecast=[ForecastPoint(**p) for p in forecast_points],
        historical_avg_weekly_demand=round(hist_avg, 2),
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/v1/forecast/bulk", response_model=BulkForecastResponse,
          tags=["Forecasting"])
def forecast_bulk_by_category(req: BulkForecastRequest):
    """
    Forecast demand for all products in a given category.

    Returns an aggregated weekly demand total for each product,
    summed across the requested horizon.

    **Example request body:**
    ```json
    {
      "category_name": "Sporting Goods",
      "horizon_days": 30
    }
    ```
    """
    bundle       = ModelRegistry.get_best_bundle()
    feature_df, prophet_df = ModelRegistry.get_feature_data()
    horizon_weeks = _horizon_weeks(req.horizon_days)

    # Filter to the requested category
    cat_mask = feature_df["category_name"] == req.category_name
    if not cat_mask.any():
        raise HTTPException(
            status_code=404,
            detail=f"Category '{req.category_name}' not found in dataset."
        )

    products = feature_df[cat_mask]["product_name"].unique().tolist()
    model        = bundle["model"]
    feature_cols = bundle.get("feature_cols", [])
    model_name   = bundle["name"]

    items = []
    for product in products:
        try:
            if model_name == "xgboost":
                pts = _xgb_product_forecast(
                    model, feature_df, feature_cols,
                    product, horizon_weeks, region=None,
                )
            else:
                pts = _prophet_product_forecast(model, prophet_df, horizon_weeks)

            total = sum(p["predicted_demand"] for p in pts)
            items.append(BulkForecastItem(
                product_name=product,
                total_predicted_demand=round(total, 2),
                forecast_weeks=horizon_weeks,
            ))
        except Exception as e:
            logger.warning("Skipping product '%s': %s", product, e)
            continue

    items.sort(key=lambda x: x.total_predicted_demand, reverse=True)

    return BulkForecastResponse(
        category_name=req.category_name,
        horizon_days=req.horizon_days,
        products=items,
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
