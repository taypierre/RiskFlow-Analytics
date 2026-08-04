import pandas as pd
import streamlit as st
import plotly.express as px
from data.market_data import fetch_commodity_data, calculate_log_returns
from models.volatility import forecast_garch_volatility
from models.risk_metrics import calculate_conditional_var

# --- UI Configuration ---
st.set_page_config(page_title="Conditional Risk Engine", layout="wide")
st.title("Conditional Risk Engine")
st.markdown("Dynamic GARCH-CVaR Forecaster for Commodity Markets")

# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "CL=F, GC=F, SI=F")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 99)

# Convert UI inputs to model variables
tickers = [t.strip() for t in tickers_input.split(",")]
alpha = 1.0 - (confidence_level / 100.0)

# --- Main Execution ---
if st.sidebar.button("Run Risk Engine"):
    with st.spinner("Fetching market data and running GARCH models..."):
        
        # 1. Fetch Data
        prices = fetch_commodity_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        returns = calculate_log_returns(prices)
        
        st.subheader("Next-Day Risk Forecasts")
        
        # Create columns for a clean layout
        cols = st.columns(len(tickers))
        
        for i, ticker in enumerate(tickers):
            asset_returns = returns[ticker].dropna()
            
            # 2. Run Math Models
            predicted_vol = forecast_garch_volatility(asset_returns)
            risk_metrics = calculate_conditional_var(asset_returns, predicted_vol, alpha=alpha)
            
            # 3. Format Outputs
            vol_pct = predicted_vol * 100
            var_pct = risk_metrics['VaR_99'] * 100
            cvar_pct = risk_metrics['CVaR_99'] * 100
            
            # 4. Render Metrics in Columns
            with cols[i]:
                st.markdown(f"### {ticker}")
                st.metric("Forecasted Volatility", f"{vol_pct:.2f}%")
                st.metric(f"VaR ({confidence_level}%)", f"{var_pct:.2f}%")
                st.metric(f"CVaR ({confidence_level}%)", f"{cvar_pct:.2f}%")
                
        # --- Interactive Charting ---
        st.markdown("---")
        st.subheader("Historical Normalized Prices")
        
        # Normalize prices to 100 for easy visual comparison
        normalized_prices = (prices / prices.iloc[0]) * 100
        fig = px.line(normalized_prices, title="Asset Performance (Base 100)", labels={'value': 'Normalized Price', 'Date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)