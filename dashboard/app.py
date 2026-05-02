"""
dashboard/app.py - Memory-optimised version
Uses SQL aggregations instead of loading the full raw DataFrame.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
PALETTE = {
    "primary":   "#2196F3",
    "secondary": "#FF5722",
    "success":   "#4CAF50",
    "warning":   "#FF9800",
    "danger":    "#F44336",
}
sns.set_theme(style="darkgrid", palette="muted")


@st.cache_data(show_spinner="Querying database …", ttl=1800)
def load_db_aggregations() -> dict:
    from src.database import (
        total_demand_by_region, demand_by_category,
        monthly_demand_trend, top_products_by_demand,
        inventory_risk_query, late_delivery_summary,
    )
    return {
        "by_region":    total_demand_by_region(),
        "by_category":  demand_by_category(),
        "monthly":      monthly_demand_trend(),
        "top_products": top_products_by_demand(30),
        "inv_risk":     inventory_risk_query(),
        "late_summary": late_delivery_summary(),
    }


def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_forecast_product(product_name, horizon_days, region=None):
    payload = {"product_name": product_name, "horizon_days": horizon_days}
    if region:
        payload["region"] = region
    try:
        r = requests.post(f"{API_BASE}/forecast/product", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.warning(f"API error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        st.warning(f"API unreachable: {e}")
    return None


def api_bulk_forecast(category, horizon_days):
    try:
        r = requests.post(f"{API_BASE}/forecast/bulk",
                          json={"category_name": category, "horizon_days": horizon_days},
                          timeout=120)
        if r.status_code == 200:
            return r.json()
        st.warning(f"API error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        st.warning(f"API unreachable: {e}")
    return None


def render_sidebar(agg):
    st.sidebar.title("📦 SC Demand Forecast")
    st.sidebar.markdown("---")
    filters = {}
    regions = ["All"] + sorted(agg["by_region"]["order_region"].unique().tolist())
    filters["region"] = st.sidebar.selectbox("Order Region", regions)
    categories = sorted(agg["by_category"]["category_name"].unique().tolist())
    filters["category"] = st.sidebar.selectbox("Product Category", categories)
    filters["horizon"] = st.sidebar.radio("Forecast Horizon", [30, 60, 90], horizontal=True)
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ API Status")
    health = api_health()
    if health:
        st.sidebar.success(f"✅ API Online — model: **{health.get('best_model','?')}**")
    else:
        st.sidebar.error("❌ API Offline")
        st.sidebar.caption("Run: python -m uvicorn api.main:app --port 8000")
    return filters


def tab_demand_overview(agg):
    st.header("📊 Demand Overview")
    monthly = agg["monthly"].copy()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])
    total_units = int(agg["by_region"]["total_quantity"].sum())
    total_rev   = float(agg["by_region"]["total_sales"].sum())
    n_products  = int(agg["top_products"].shape[0])
    avg_late    = float(agg["late_summary"]["late_rate_pct"].mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Units Ordered", f"{total_units:,}")
    k2.metric("Total Revenue", f"${total_rev:,.0f}")
    k3.metric("Unique Products", f"{n_products}")
    k4.metric("Avg Late Delivery %", f"{avg_late:.1f}%")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📅 Monthly Demand Trend")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(monthly["year_month"], monthly["total_quantity"],
                color=PALETTE["primary"], linewidth=2, marker="o", markersize=4)
        ax.fill_between(monthly["year_month"], monthly["total_quantity"],
                        alpha=0.15, color=PALETTE["primary"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Units")
        ax.set_title("Monthly Order Quantity", fontweight="bold")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("🌍 Demand by Region")
        region_df = agg["by_region"].sort_values("total_quantity", ascending=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.barh(region_df["order_region"], region_df["total_quantity"],
                color=PALETTE["primary"], alpha=0.85)
        ax.set_xlabel("Units")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🏷 Revenue by Category")
        cat_df = agg["by_category"].sort_values("total_sales", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(cat_df["category_name"], cat_df["total_sales"],
               color=sns.color_palette("Blues_r", len(cat_df)))
        ax.set_ylabel("Revenue ($)")
        plt.xticks(rotation=40, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.subheader("🏆 Top 15 Products by Volume")
        tp = agg["top_products"].head(15)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(tp["product_name"], tp["total_quantity"],
                color=PALETTE["secondary"], alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Units")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def tab_forecast(agg, filters):
    st.header("🔮 Demand Forecast")
    products = sorted(agg["top_products"]["product_name"].unique().tolist())
    product  = st.selectbox("Select Product", products, key="fc_product")
    horizon  = filters["horizon"]
    region   = filters["region"] if filters["region"] != "All" else None

    col_live, col_hist = st.columns(2)

    with col_live:
        st.subheader(f"📡 Live API Forecast — {horizon} days")
        if st.button("Fetch Live Forecast"):
            with st.spinner("Calling API …"):
                result = api_forecast_product(product, horizon, region)
            if result:
                fc_df = pd.DataFrame(result["forecast"])
                fc_df["date"] = pd.to_datetime(fc_df["date"])
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(fc_df["date"], fc_df["predicted_demand"],
                        color=PALETTE["secondary"], linewidth=2, marker="o",
                        markersize=4, label="Forecast")
                ax.fill_between(fc_df["date"], fc_df["lower_bound"], fc_df["upper_bound"],
                                alpha=0.2, color=PALETTE["secondary"], label="90% CI")
                ax.set_xlabel("Week")
                ax.set_ylabel("Predicted Demand (units)")
                ax.set_title(f"{product} — {horizon}-day Forecast", fontweight="bold")
                ax.legend()
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(f"Model: **{result['model_used']}** | "
                           f"Hist. avg: **{result['historical_avg_weekly_demand']} units/wk**")
                st.dataframe(fc_df.rename(columns={
                    "date": "Week", "predicted_demand": "Predicted",
                    "lower_bound": "Lower", "upper_bound": "Upper",
                }), use_container_width=True)
            else:
                st.info("Start the FastAPI server to enable live forecasts.")

    with col_hist:
        st.subheader("📈 Historical Monthly Demand")
        monthly = agg["monthly"].copy()
        monthly["year_month"] = pd.to_datetime(monthly["year_month"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(monthly["year_month"], monthly["total_quantity"],
                color=PALETTE["primary"], linewidth=1.8)
        ax.fill_between(monthly["year_month"], monthly["total_quantity"],
                        alpha=0.12, color=PALETTE["primary"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Weekly Demand")
        ax.set_title("Overall Market Demand Trend", fontweight="bold")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader(f"📦 Bulk Category Forecast — {filters['category']}")
    if st.button("Run Bulk Forecast"):
        with st.spinner("Fetching bulk forecast …"):
            bulk = api_bulk_forecast(filters["category"], horizon)
        if bulk:
            bulk_df = pd.DataFrame([{
                "Product": p["product_name"],
                f"Total Demand ({horizon}d)": p["total_predicted_demand"],
                "Weeks": p["forecast_weeks"],
            } for p in bulk["products"]])
            st.dataframe(bulk_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.barh(bulk_df["Product"], bulk_df[f"Total Demand ({horizon}d)"],
                    color=PALETTE["primary"], alpha=0.85)
            ax.invert_yaxis()
            ax.set_xlabel(f"Total Predicted Demand ({horizon} days)")
            ax.set_title(f"Category: {filters['category']} — Bulk Forecast", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Start the FastAPI server to enable bulk forecasts.")


def tab_inventory_risk(agg):
    st.header("⚠️ Inventory & Fulfilment Risk")
    inv_risk  = agg["inv_risk"]
    late_summ = agg["late_summary"]

    high_risk = inv_risk[inv_risk["avg_delay"] > 1]
    r1, r2, r3 = st.columns(3)
    r1.metric("Products with Avg Delay > 1 day", len(high_risk))
    r2.metric("Max Avg Delay (days)", f"{inv_risk['avg_delay'].max():.2f}" if len(inv_risk) else "—")
    r3.metric("Highest Late-Delivery Rate", f"{late_summ['late_rate_pct'].max():.1f}%" if len(late_summ) else "—")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚚 Late Delivery Rate by Mode & Region")
        pivot = late_summ.pivot_table(
            index="shipping_mode", columns="order_region",
            values="late_rate_pct", aggfunc="mean"
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Late Rate %"})
        ax.set_title("Late Delivery Rate (%) — Mode × Region", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("🔴 Most Delayed Products")
        top_delayed = inv_risk.head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [PALETTE["danger"] if d > 2 else PALETTE["warning"]
                  for d in top_delayed["avg_delay"]]
        ax.barh(top_delayed["product_name"], top_delayed["avg_delay"],
                color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Avg Delay (days)")
        ax.set_title("Top 15 Products by Avg Shipping Delay", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📋 Full Risk Table")
    st.dataframe(inv_risk.rename(columns={
        "product_name": "Product", "category_name": "Category",
        "avg_scheduled_days": "Scheduled", "avg_actual_days": "Actual",
        "avg_delay": "Avg Delay", "order_count": "Orders",
    }), use_container_width=True, height=400)

    critical = inv_risk[inv_risk["avg_delay"] > 2]
    if len(critical) > 0:
        st.error(f"🚨 **{len(critical)} products** have avg shipping delay > 2 days.")


def tab_data_explorer(agg):
    st.header("🔬 Data Explorer")
    st.subheader("Demand by Region")
    st.dataframe(agg["by_region"], use_container_width=True)
    st.subheader("Demand by Category")
    st.dataframe(agg["by_category"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        cat_df = agg["by_category"].sort_values("total_quantity", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.pie(cat_df["total_quantity"], labels=cat_df["category_name"],
               autopct="%1.1f%%", startangle=140,
               colors=sns.color_palette("pastel", len(cat_df)))
        ax.set_title("Demand Share by Category (Top 8)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.subheader("Top 20 Products")
        st.dataframe(agg["top_products"].head(20), use_container_width=True)

    st.subheader("Monthly Trend")
    st.dataframe(agg["monthly"], use_container_width=True)


def main():
    st.markdown("<h1 style='text-align:center;color:#2196F3;'>📦 Supply Chain Demand Forecasting</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:grey;'>DataCo Smart Supply Chain | XGBoost + Prophet | Interactive Analytics Dashboard</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    try:
        agg = load_db_aggregations()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.info("Run `python src/data_loader.py` first to populate the database.")
        st.stop()

    filters = render_sidebar(agg)

    tabs = st.tabs(["📊 Demand Overview", "🔮 Forecast", "⚠️ Inventory Risk", "🔬 Data Explorer"])
    with tabs[0]:
        tab_demand_overview(agg)
    with tabs[1]:
        tab_forecast(agg, filters)
    with tabs[2]:
        tab_inventory_risk(agg)
    with tabs[3]:
        tab_data_explorer(agg)

    st.markdown("---")
    st.caption("Supply Chain Demand Forecasting | Streamlit + FastAPI + XGBoost + Prophet")


if __name__ == "__main__":
    main()
