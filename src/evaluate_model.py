"""
evaluate_model.py
-----------------
Computes forecast evaluation metrics and produces diagnostic plots.

Metrics
-------
- MAE  : Mean Absolute Error
- RMSE : Root Mean Squared Error
- MAPE : Mean Absolute Percentage Error (ignores zero-actual rows)

Plots saved to reports/
------------------------
- forecast_vs_actual.png
- residuals_distribution.png
- error_by_product.png
- feature_importance.png  (XGBoost only)
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
REPORTS  = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = {"actual": "#2196F3", "predicted": "#FF5722", "residual": "#4CAF50"}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray,
                    predicted: np.ndarray,
                    label: str = "Model") -> dict:
    """
    Compute MAE, RMSE, and MAPE.

    Parameters
    ----------
    actual : array-like
        Ground truth demand values.
    predicted : array-like
        Model predictions (clipped to ≥ 0 before computation).
    label : str
        Name tag for logging.

    Returns
    -------
    dict with keys: MAE, RMSE, MAPE, label
    """
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float).clip(0)

    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    # MAPE: exclude rows where actual == 0 to avoid division by zero
    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(
            np.abs((actual[nonzero_mask] - predicted[nonzero_mask])
                   / actual[nonzero_mask])
        ) * 100)
    else:
        mape = float("nan")

    metrics = {"label": label, "MAE": round(mae, 4),
               "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}
    logger.info("[%s] MAE=%.4f | RMSE=%.4f | MAPE=%.2f%%",
                label, mae, rmse, mape)
    return metrics


def print_metrics_table(metrics_list: list[dict]) -> None:
    """Pretty-print a comparison table to stdout."""
    header = f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(f"{m['label']:<20} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} "
              f"{m['MAPE']:>10.2f}")
    print("=" * len(header) + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_forecast_vs_actual(actual: np.ndarray,
                             predicted: np.ndarray,
                             label: str = "Forecast vs Actual",
                             save_path: Path | None = None) -> Path:
    """
    Line chart of actual vs predicted demand over the test period.

    Parameters
    ----------
    actual : array-like
    predicted : array-like
    label : str
    save_path : Path or None
        If None, saves to reports/forecast_vs_actual.png.

    Returns
    -------
    Path to saved figure.
    """
    if save_path is None:
        save_path = REPORTS / "forecast_vs_actual.png"

    actual    = np.array(actual)
    predicted = np.array(predicted).clip(0)
    x         = np.arange(len(actual))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, actual,    label="Actual",    color=COLORS["actual"],
            linewidth=1.8, alpha=0.9)
    ax.plot(x, predicted, label="Predicted", color=COLORS["predicted"],
            linewidth=1.8, linestyle="--", alpha=0.9)
    ax.fill_between(x, actual, predicted, alpha=0.12, color=COLORS["residual"])
    ax.set_title(label, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Demand (units)")
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


def plot_residuals(actual: np.ndarray,
                   predicted: np.ndarray,
                   save_path: Path | None = None) -> Path:
    """
    Two-panel plot: residual scatter + residual distribution histogram.
    """
    if save_path is None:
        save_path = REPORTS / "residuals_distribution.png"

    actual    = np.array(actual)
    predicted = np.array(predicted).clip(0)
    residuals = actual - predicted

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # Scatter
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(predicted, residuals, alpha=0.35, s=15,
                color=COLORS["residual"], edgecolors="none")
    ax1.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax1.set_xlabel("Predicted Demand")
    ax1.set_ylabel("Residual (Actual − Predicted)")
    ax1.set_title("Residuals vs Fitted", fontweight="bold")

    # Histogram
    ax2 = fig.add_subplot(gs[1])
    sns.histplot(residuals, bins=40, kde=True, ax=ax2,
                 color=COLORS["residual"], alpha=0.7)
    ax2.axvline(0, color="red", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Residual")
    ax2.set_title("Residual Distribution", fontweight="bold")

    plt.suptitle("Residual Diagnostics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


def plot_error_by_product(results_df: pd.DataFrame,
                           top_n: int = 20,
                           save_path: Path | None = None) -> Path:
    """
    Bar chart of RMSE per product for the top *top_n* highest-volume products.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns: product_name, demand, predicted.
    top_n : int
    save_path : Path or None
    """
    if save_path is None:
        save_path = REPORTS / "error_by_product.png"

    grp = results_df.groupby("product_name").apply(
        lambda g: pd.Series({
            "rmse":   float(np.sqrt(np.mean((g["demand"] - g["predicted"]) ** 2))),
            "volume": g["demand"].sum(),
        })
    ).reset_index()

    # Focus on high-volume products
    top_products = grp.nlargest(top_n, "volume")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(
        top_products["product_name"],
        top_products["rmse"],
        color=COLORS["predicted"],
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_xlabel("RMSE (units)")
    ax.set_title(f"Forecast RMSE — Top {top_n} Products by Volume",
                 fontweight="bold", pad=10)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


def plot_feature_importance(model,
                             feature_cols: list[str],
                             top_n: int = 25,
                             save_path: Path | None = None) -> Path:
    """
    Horizontal bar chart of XGBoost feature importances (gain).

    Parameters
    ----------
    model : xgb.XGBRegressor
    feature_cols : list[str]
    top_n : int
    save_path : Path or None
    """
    if save_path is None:
        save_path = REPORTS / "feature_importance.png"

    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(importance.items(), columns=["feature", "importance"])
          .sort_values("importance", ascending=False)
          .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=imp_df, x="importance", y="feature",
                palette="Blues_r", ax=ax)
    ax.set_title(f"XGBoost Feature Importance (Gain) — Top {top_n}",
                 fontweight="bold", pad=10)
    ax.set_xlabel("Gain")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


def plot_prophet_components(model, forecast: pd.DataFrame,
                             save_path: Path | None = None) -> Path:
    """
    Save Prophet's built-in component decomposition plot.
    """
    if save_path is None:
        save_path = REPORTS / "prophet_components.png"

    fig = model.plot_components(forecast)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


def plot_metrics_comparison(metrics_list: list[dict],
                              save_path: Path | None = None) -> Path:
    """
    Grouped bar chart comparing MAE / RMSE / MAPE across models.
    """
    if save_path is None:
        save_path = REPORTS / "metrics_comparison.png"

    metric_names = ["MAE", "RMSE", "MAPE"]
    models = [m["label"] for m in metrics_list]
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, m in enumerate(metrics_list):
        vals = [m.get(k, 0) for k in metric_names]
        ax.bar(x + i * width, vals, width,
               label=m["label"], color=palette[i % len(palette)], alpha=0.85)

    ax.set_xticks(x + width * (len(metrics_list) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Error Value")
    ax.set_title("Model Comparison — Error Metrics", fontweight="bold", pad=10)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ── Full evaluation run ───────────────────────────────────────────────────────

def run_evaluation():
    """
    Load saved models, run predictions on the test split,
    compute all metrics, and produce all diagnostic plots.
    """
    import sys
    import pickle
    sys.path.insert(0, str(ROOT))
    from src.data_loader import load_raw
    from src.preprocessing import preprocess
    from src.feature_engineering import build_feature_matrix
    from src.train_model import (time_split, get_feature_cols,
                                  predict_xgboost, predict_prophet,
                                  load_xgboost, load_prophet)

    logger.info("=== Running Evaluation ===")

    raw     = load_raw()
    clean   = preprocess(raw)
    feat_df, prophet_df = build_feature_matrix(clean)
    feature_cols = get_feature_cols(feat_df)
    _, test = time_split(feat_df)

    # XGBoost
    xgb_model   = load_xgboost()
    xgb_results = predict_xgboost(xgb_model, test, feature_cols)
    xgb_metrics = compute_metrics(xgb_results["demand"].values,
                                   xgb_results["predicted"].values,
                                   label="XGBoost")

    # Prophet
    prophet_model    = load_prophet()
    prophet_forecast = predict_prophet(prophet_model, 12, prophet_df)
    test_dates       = test["week_start"].unique()
    p_test = prophet_forecast[
        prophet_forecast["ds"].isin(pd.to_datetime(test_dates))
    ][["ds", "yhat"]]
    actual_agg = (
        test.groupby("week_start")["demand"].sum().reset_index()
    )
    actual_agg["week_start"] = pd.to_datetime(actual_agg["week_start"])
    merged = actual_agg.merge(p_test, left_on="week_start", right_on="ds", how="inner")
    prophet_metrics = compute_metrics(merged["demand"].values,
                                       merged["yhat"].clip(0).values,
                                       label="Prophet")

    all_metrics = [xgb_metrics, prophet_metrics]
    print_metrics_table(all_metrics)

    # Plots
    plot_forecast_vs_actual(xgb_results["demand"].values,
                             xgb_results["predicted"].values,
                             label="XGBoost — Forecast vs Actual (Test Set)")
    plot_residuals(xgb_results["demand"].values, xgb_results["predicted"].values)
    plot_error_by_product(xgb_results)
    plot_feature_importance(xgb_model, feature_cols)
    plot_metrics_comparison(all_metrics)

    logger.info("=== Evaluation Complete. Charts saved to reports/ ===")
    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    run_evaluation()
