import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data.market_data import fetch_commodity_data, calculate_log_returns
from models.volatility import forecast_garch_volatility
from models.risk_metrics import calculate_conditional_var
from models.back_testing import run_historical_backtest, kupiec_pof_test

# --- UI Configuration ---
st.set_page_config(page_title="RiskFlow Analytics", layout="wide")
st.title("RiskFlow Analytics")
st.markdown("Dynamic GARCH-CVaR Forecaster for Commodity Markets")

# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "CL=F, GC=F, SI=F")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 99)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Validation")
run_backtest = st.sidebar.checkbox("Run Historical Backtest (Kupiec POF)")
st.sidebar.caption("Note: Running the backtest calculates hundreds of rolling GARCH models and may take a moment.")

tickers = [t.strip() for t in tickers_input.split(",")]
alpha = 1.0 - (confidence_level / 100.0)

if st.sidebar.button("Run Risk Engine"):
    with st.spinner("Fetching market data and calculating risk metrics..."):
        
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

        # --- Backtesting & Validation UI ---
        if run_backtest:
            st.markdown("---")
            st.subheader("Model Validation: Kupiec Proportion of Failures (POF)")
            
            # Create tabs for each ticker to keep the UI uncluttered
            tabs = st.tabs(tickers)
            
            for i, ticker in enumerate(tickers):
                with tabs[i]:
                    asset_returns = returns[ticker].dropna()
                    
                    with st.spinner(f"Running rolling window backtest for {ticker}..."):
                        bt_results, actual_rate, expected_rate, breaches, total_days = run_historical_backtest(
                            asset_returns, window_size=252, alpha=alpha
                        )
                        
                        model_passed, lr, critical_value = kupiec_pof_test(
                            expected_rate, breaches, total_days, confidence_level=(confidence_level/100.0)
                        )
                        
                        if model_passed:
                            st.success(f"✅ **MODEL PASSED:** The {ticker} engine successfully predicted risk within the {confidence_level}% confidence interval.")
                        else:
                            st.error(f"🚨 **MODEL FAILED:** The {ticker} engine breached the VaR threshold too frequently. The Likelihood Ratio exceeded the critical cutoff.")
                            
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Days Tested", total_days)
                        m2.metric("Expected Breaches", f"{expected_rate * total_days:.1f}")
                        m3.metric("Actual Breaches", breaches)
                        
                        m4.metric("Likelihood Ratio (LR)", f"{lr:.2f}", delta=f"Cutoff: {critical_value:.2f}", delta_color="inverse")
                        
                        st.markdown(f"**{ticker} Historical Returns vs. Predicted VaR**")
                        
                        fig_bt = go.Figure()
                        
                        fig_bt.add_trace(go.Scatter(x=bt_results['Date'], y=bt_results['Actual_Return'], 
                                                    mode='lines', name='Actual Return', line=dict(color='blue', width=1)))
                        
                        fig_bt.add_trace(go.Scatter(x=bt_results['Date'], y=bt_results['Predicted_VaR'], 
                                                    mode='lines', name=f'Predicted VaR ({confidence_level}%)', line=dict(color='red', width=2)))
                        
                        breach_points = bt_results[bt_results['Breach'] == True]
                        fig_bt.add_trace(go.Scatter(x=breach_points['Date'], y=breach_points['Actual_Return'], 
                                                    mode='markers', name='Breach (Loss > VaR)', 
                                                    marker=dict(color='red', size=8, symbol='x')))
                        
                        fig_bt.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_bt, use_container_width=True)