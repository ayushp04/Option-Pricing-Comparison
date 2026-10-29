import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Option Pricing Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("Option Pricing Model Comparison")
st.caption("Black–Scholes vs Bachelier | January 2023 – December 2025")

# Load previously calculated results
data_path = Path(__file__).parent / "dashboard_data.pkl"

if not data_path.exists():
    st.error("dashboard_data.pkl was not found. Place it beside app.py.")
    st.stop()

with open(data_path, "rb") as f:
    data = pickle.load(f)

results_all = data["results_all"].copy()
overall_metrics = data["overall_metrics"].copy()
monthly_metrics = data["monthly_metrics"].copy()
win_summary = data["win_summary"].copy()


# Sticky header with stock selector
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
    overflow: visible;
}

.sticky-stock-header {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: var(--background-color, white);
    padding: 0.6rem 0 0.8rem 0;
    border-bottom: 1px solid rgba(128, 128, 128, 0.3);
}
</style>
""", unsafe_allow_html=True)

stocks = sorted(results_all["Stock"].dropna().unique())

st.markdown('<div class="sticky-stock-header">', unsafe_allow_html=True)
selected_stock = st.selectbox(
    "Select stock",
    stocks,
    key="global_stock_selector"
)
st.markdown('</div>', unsafe_allow_html=True)

stock_data = results_all[
    results_all["Stock"] == selected_stock
].copy()

# Overall performance summary
st.header("1. Overall Performance")

overall_stock = overall_metrics[
    overall_metrics["Stock"] == selected_stock
].sort_values("MAE")

best = overall_stock.iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Best configuration (MAE)", best["Configuration"])
c2.metric("Lowest overall MAE", f"₹{best['MAE']:,.2f}")
c3.metric("Lowest overall WAPE", f"{overall_stock['WAPE (%)'].min():.2f}%")

st.dataframe(
    overall_stock[
        ["Configuration", "Options", "MAE", "WAPE (%)"]
    ].round(3),
    use_container_width=True,
    hide_index=True
)

fig_overall = px.bar(
    overall_stock,
    x="Configuration",
    y="MAE",
    color="Configuration",
    title=f"Overall MAE — {selected_stock}",
    labels={"MAE": "Mean Absolute Error (₹)"}
)
st.plotly_chart(fig_overall, use_container_width=True)

# Daily analysis
st.header("2. Daily Pricing Analysis")

dates = sorted(stock_data["Valuation Date"].dropna().unique())

if len(dates) == 0:
    st.warning("No daily observations are available for this stock.")
else:
    selected_date = st.selectbox(
        "Select valuation date",
        dates,
        format_func=lambda d: pd.Timestamp(d).strftime("%d %B %Y")
    )

    daily = stock_data[
        stock_data["Valuation Date"] == selected_date
    ].copy()

    daily_summary = (
        daily.groupby("Configuration")
        .agg(
            Options=("Absolute Error", "size"),
            MAE=("Absolute Error", "mean"),
            Total_Error=("Absolute Error", "sum"),
            Total_Premium=("Market Price", "sum")
        )
        .reset_index()
    )

    daily_summary["WAPE (%)"] = (
        100 * daily_summary["Total_Error"]
        / daily_summary["Total_Premium"].replace(0, float("nan"))
    )

    st.dataframe(
        daily_summary[
            ["Configuration", "Options", "MAE", "WAPE (%)"]
        ].round(3),
        use_container_width=True,
        hide_index=True
    )

    daily_plot = daily.melt(
        id_vars=["Expiry", "Strike", "Market Price", "Configuration"],
        value_vars=["Model Price"],
        var_name="Price Type",
        value_name="Price"
    )

    st.caption(
        "The chart compares model prices with observed market premiums "
        "for the selected date."
    )

    compare = daily[
        ["Expiry", "Strike", "Market Price", "Configuration", "Model Price"]
    ].melt(
        id_vars=["Expiry", "Strike", "Configuration"],
        value_vars=["Market Price", "Model Price"],
        var_name="Price Type",
        value_name="Premium (₹)"
    )

    compare["Option"] = (
        "Strike ₹" + compare["Strike"].round(2).astype(str)
        + " | Expiry "
        + pd.to_datetime(compare["Expiry"]).dt.strftime("%d-%b-%Y")
    )

    fig_daily = px.bar(
        compare,
        x="Option",
        y="Premium (₹)",
        color="Price Type",
        barmode="group",
        facet_col="Configuration",
        facet_col_wrap=2,
        title="Market Premium vs Model Price",
        height=650
    )
    fig_daily.update_layout(showlegend=True)
    st.plotly_chart(fig_daily, use_container_width=True)


# Monthly analysis
st.header("3. Monthly Comparison")

monthly_stock = monthly_metrics[
    monthly_metrics["Stock"] == selected_stock
].copy()

months = sorted(monthly_stock["Month"].dropna().unique())

selected_month = st.selectbox(
    "Select month",
    ["All months"] + months,
    key="monthly_section_selector"
)

if selected_month != "All months":
    monthly_stock = monthly_stock[
        monthly_stock["Month"] == selected_month
    ]

fig_monthly = px.line(
    monthly_stock,
    x="Month",
    y="MAE",
    color="Configuration",
    markers=True,
    title="Monthly MAE by Configuration",
    labels={"MAE": "Mean Absolute Error (₹)"}
)
st.plotly_chart(fig_monthly, use_container_width=True)

fig_wape = px.line(
    monthly_stock,
    x="Month",
    y="WAPE (%)",
    color="Configuration",
    markers=True,
    title="Monthly WAPE by Configuration"
)
st.plotly_chart(fig_wape, use_container_width=True)

st.dataframe(
    monthly_stock[
        ["Month", "Configuration", "Options", "MAE", "WAPE (%)"]
    ].sort_values(["Month", "MAE"]).round(3),
    use_container_width=True,
    hide_index=True
)


monthly_stock = monthly_metrics[
    monthly_metrics["Stock"] == selected_stock
].copy()

if selected_month != "All months":
    monthly_stock = monthly_stock[
        monthly_stock["Month"] == selected_month
    ]

fig_monthly = px.line(
    monthly_stock,
    x="Month",
    y="MAE",
    color="Configuration",
    markers=True,
    title="Monthly MAE by Configuration",
    labels={"MAE": "Mean Absolute Error (₹)"}
)
st.plotly_chart(
    fig_monthly,
    use_container_width=True,
    key="monthly_mae_chart"
)

fig_wape = px.line(
    monthly_stock,
    x="Month",
    y="WAPE (%)",
    color="Configuration",
    markers=True,
    title="Monthly WAPE by Configuration"
)
st.plotly_chart(
    fig_wape,
    use_container_width=True,
    key="monthly_wape_chart"
)

st.dataframe(
    monthly_stock[
        ["Month", "Configuration", "Options", "MAE", "WAPE (%)"]
    ].sort_values(["Month", "MAE"]).round(3),
    use_container_width=True,
    hide_index=True
)

# Monthly win summary
st.header("4. Monthly Wins")

stock_wins = win_summary[win_summary["Stock"] == selected_stock].copy()

st.dataframe(
    stock_wins.sort_values("Months Won (MAE)", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "MAE and WAPE measure historical pricing accuracy. "
    "Lower values indicate smaller pricing errors."
)
