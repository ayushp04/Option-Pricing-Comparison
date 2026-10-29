# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:10:14 2026

@author: Ayush
"""

import pandas as pd
from pathlib import Path
from pathlib import Path
import pickle
import plotly.express as px
import pandas as pd
import streamlit as st
import numpy as np

# folder = Path(r"C:\Users\Ayush\Documents\Research Papers")

# stock_prices = pd.read_csv(
#     folder / "stock_price_history.csv",
#     parse_dates=["Date"]
# )

folder = Path(__file__).parent

stock_prices = pd.read_csv(
    folder / "stock_price_history.csv",
    parse_dates=["Date"]
)

# --------------------------------------------------
# Rolling annualized volatility of log returns
# --------------------------------------------------


trading_days_per_year = 252
window = 30


# --------------------------------------------------
# Calculate 30-day rolling annualized volatility
# --------------------------------------------------
import numpy as np

volatility_data = stock_prices.sort_values(
    ["Stock", "Date"]
).copy()

# Daily log returns, calculated separately for each stock
volatility_data["log_return"] = (
    volatility_data.groupby("Stock")["Price"]
    .transform(lambda prices: np.log(prices / prices.shift(1)))
)

# Annualized rolling volatility, expressed as a percentage
volatility_data["rolling_vol"] = (
    volatility_data.groupby("Stock")["log_return"]
    .transform(
        lambda returns: returns.rolling(30).std()
        * np.sqrt(252)
        * 100
    )
)


# --------------------------------------------------
# 1. Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Option Pricing Analysis",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# 2. File paths
# --------------------------------------------------
FOLDER = Path(r"C:\Users\Ayush\Documents\Research Papers")

# --------------------------------------------------
# 3. Load option-pricing results
# --------------------------------------------------
with open(FOLDER / "dashboard_data.pkl", "rb") as file:
    saved_data = pickle.load(file)

results_all = saved_data["results_all"]

# --------------------------------------------------
# 4. Load historical stock prices
# --------------------------------------------------
stock_prices = pd.read_csv(
    FOLDER / "stock_price_history.csv",
    parse_dates=["Date"]
)

# Remove any timezone information, if present
stock_prices["Date"] = (
    pd.to_datetime(stock_prices["Date"]).dt.tz_localize(None)
)

# --------------------------------------------------
# 5. Dashboard title
# --------------------------------------------------
st.title("Option Pricing Analysis")
st.caption(
    "Historical comparison of Black–Scholes and Bachelier "
    "call-option pricing models"
)

st.divider()

# --------------------------------------------------
# 6. Daily pricing analysis selectors
# --------------------------------------------------
st.header("1. Daily Pricing Analysis")

available_stocks = sorted(results_all["Stock"].dropna().unique())

selected_stock = st.selectbox(
    "Select stock",
    available_stocks
)

stock_results = results_all[
    results_all["Stock"] == selected_stock
].copy()

available_dates = sorted(
    pd.to_datetime(stock_results["Valuation Date"])
    .dt.strftime("%Y-%m-%d")
    .unique(),
    reverse=True
)

selected_date = st.selectbox(
    "Select valuation date",
    available_dates
)


st.write(f"**Selected stock:** {selected_stock}")
st.write(f"**Valuation date:** {selected_date}")


# Find the stock price on the selected valuation date
valuation_date = pd.Timestamp(selected_date)

price_on_date = stock_prices[
    (stock_prices["Stock"] == selected_stock)
    & (stock_prices["Date"] == valuation_date)
]

if not price_on_date.empty:
    spot_price = float(price_on_date["Price"].iloc[0])
    st.metric(
        label=f"{selected_stock} closing price on {selected_date}",
        value=f"₹{spot_price:,.2f}"
    )
else:
    st.warning("No stock-price data found for this valuation date.")


# --------------------------------------------------
# Volatility estimates used for option pricing
# --------------------------------------------------
st.subheader("Volatility Estimates for Selected Date")

# Filter pricing results for the selected stock and date
daily_vol = results_all[
    (results_all["Stock"] == selected_stock)
    & (
        pd.to_datetime(results_all["Valuation Date"]).dt.strftime("%Y-%m-%d")
        == selected_date
    )
].copy()

if daily_vol.empty:
    st.warning("No volatility estimates found for this date.")
else:
    # These estimates should be the same across option rows for the date
    bs_60 = daily_vol["BS Vol 60-day"].dropna()
    bs_2y = daily_vol["BS Vol 2-year"].dropna()
    bach_60 = daily_vol["Bachelier Vol 60-day"].dropna()
    bach_2y = daily_vol["Bachelier Vol 2-year"].dropna()

    # Display four metrics in one row
    c1, c2, c3, c4 = st.columns(4)

    if not bs_60.empty:
        c1.metric("Black–Scholes · 60-day", f"{bs_60.iloc[0] * 100:.2f}%")
    if not bs_2y.empty:
        c2.metric("Black–Scholes · 2-year", f"{bs_2y.iloc[0] * 100:.2f}%")
    if not bach_60.empty:
        c3.metric("Bachelier · 60-day", f"₹{bach_60.iloc[0]:.2f}")
    if not bach_2y.empty:
        c4.metric("Bachelier · 2-year", f"₹{bach_2y.iloc[0]:.2f}")

    st.caption(
        "Black–Scholes volatility is shown as annualized percentage volatility. "
        "Bachelier volatility is shown in price units per square-root year (₹/√year)."
    )


# --------------------------------------------------
# 7. Check available option contracts
# --------------------------------------------------
st.subheader("Selected-date option contracts")

daily_results = stock_results[
    pd.to_datetime(stock_results["Valuation Date"]).dt.strftime("%Y-%m-%d")
    == selected_date
].copy()

contract_columns = [
    "Expiry",
    "Strike",
    "Market Price",
    "Configuration",
    "Model Price",
    "Absolute Error"
]

st.write(
    f"Number of pricing records for this date: {len(daily_results)}"
)


# Round displayed prices and errors to 2 decimal places
display_results = daily_results[contract_columns].copy()

price_columns = ["Market Price", "Model Price", "Absolute Error"]

display_results[price_columns] = display_results[price_columns].round(2)

st.dataframe(
    display_results.sort_values(
        ["Expiry", "Strike", "Configuration"]
    ),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 8. Historical stock-price charts
# --------------------------------------------------
st.subheader("Historical Closing Prices")

# Convert selected date to a date object
valuation_date = pd.Timestamp(selected_date)

# Filter prices for the selected stock up to the valuation date
historical_prices = stock_prices[
    (stock_prices["Stock"] == selected_stock)
    & (stock_prices["Date"] <= valuation_date)
].sort_values("Date").copy()

if historical_prices.empty:
    st.warning("No historical stock-price data available.")
else:
    # Last 60 trading observations, including the selected date
    prices_60 = historical_prices.tail(60)

    # Previous 2 calendar years, including the selected date
    start_2_years = valuation_date - pd.DateOffset(years=2)

    prices_2_years = historical_prices[
        historical_prices["Date"] >= start_2_years
    ]

    # Two charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Previous 60 Trading Days")

        fig_60 = px.line(
            prices_60,
            x="Date",
            y="Price",
            title=f"{selected_stock}: 60 Trading Days",
            labels={
                "Date": "Date",
                "Price": "Adjusted Closing Price (₹)"
            }
        )

        fig_60.update_layout(
            xaxis_title="Date",
            yaxis_title="Closing Price (₹)",
            showlegend=False
        )

        st.plotly_chart(fig_60, use_container_width=True, key="price_chart_60")

    with col2:
        st.markdown("#### Previous 2 Years")

        fig_2_years = px.line(
            prices_2_years,
            x="Date",
            y="Price",
            title=f"{selected_stock}: 2-Year Price History",
            labels={
                "Date": "Date",
                "Price": "Adjusted Closing Price (₹)"
            }
        )

        fig_2_years.update_layout(
            xaxis_title="Date",
            yaxis_title="Closing Price (₹)",
            showlegend=False
        )

        st.plotly_chart(
            fig_2_years,
            use_container_width=True,
            key="price_chart_2_years"
        )



# --------------------------------------------------
# 9. Rolling volatility: 60 days vs 2 years
# --------------------------------------------------
st.subheader("Rolling Annualized Volatility of Log Returns")
col1, col2 = st.columns(2)
valuation_date = pd.Timestamp(selected_date)

# Keep observations up to the selected valuation date
vol_history = volatility_data[
    (volatility_data["Stock"] == selected_stock)
    & (volatility_data["Date"] <= valuation_date)
].dropna(subset=["rolling_vol"]).copy()


# Define the two display periods
history_60 = vol_history.tail(60)

start_2_years = valuation_date - pd.DateOffset(years=2)

history_2_years = vol_history[
    vol_history["Date"] >= start_2_years
].copy()


color_map = {
    "Adani Enterprises": "#e6550d",
    "Hindustan Unilever": "#31a354"
}

with col1:
    st.markdown("#### Last 60 Trading Observations")

    if history_60.empty:
        st.warning("Not enough data for the 60-day chart.")
    else:
        fig_60_vol = px.line(
            history_60,
            x="Date",
            y="rolling_vol",
            color="Stock",
            color_discrete_map=color_map,
            labels={
                "Date": "Date",
                "rolling_vol": "Annualized Volatility (%)",
                "Stock": "Stock"
            }
        )

        fig_60_vol.update_layout(
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            hovermode="x unified",
            legend_title="Stock"
        )

        st.plotly_chart(
            fig_60_vol,
            use_container_width=True,
            key="rolling_vol_60_days"
        )

with col2:
    st.markdown("#### Previous 2 Years")

    if history_2_years.empty:
        st.warning("Not enough data for the 2-year chart.")
    else:
        fig_2_years_vol = px.line(
            history_2_years,
            x="Date",
            y="rolling_vol",
            color="Stock",
            color_discrete_map=color_map,
            labels={
                "Date": "Date",
                "rolling_vol": "Annualized Volatility (%)",
                "Stock": "Stock"
            }
        )

        fig_2_years_vol.update_layout(
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            hovermode="x unified",
            legend_title="Stock"
        )


        st.plotly_chart(
            fig_2_years_vol,
            use_container_width=True,
            key="rolling_vol_2_years"
        )
