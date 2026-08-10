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

# --- Sidebar Inputs Wrapped in a Form ---
with st.sidebar:
    with st.form(key="risk_form"):
        st.header("Model Parameters")
        tickers_input = st.text_input("Tickers (comma separated)", "CL=F, GC=F, SI=F")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 99)

        st.markdown("---")
        st.subheader("Portfolio Allocation")
        st.caption("Allocate weights across your selected assets.")

        tickers_temp = [t.strip() for t in tickers_input.split(",") if t.strip()]
        
        raw_weights = []
        for ticker in tickers_temp:
            weight = st.slider(f"{ticker} Weight", 0, 100, int(100 / max(len(tickers_temp), 1)))
            raw_weights.append(weight)
        
        temp_total = sum(raw_weights)
        if temp_total > 0:
            real_pcts = [f"{ticker}: {(w/temp_total)*100:.1f}%" for ticker, w in zip(tickers_temp, raw_weights)]
            st.caption(f"**Actual Allocation:** {', '.join(real_pcts)}")
        else:
            st.caption("**Actual Allocation:** 0%")

        submit_button = st.form_submit_button(label="Run Risk Engine")

    st.markdown("---")
    st.subheader("Model Validation")
    run_backtest = st.toggle("Run Historical Backtest (Kupiec POF)")
    st.caption("Note: Running the backtest calculates hundreds of rolling GARCH models and may take a moment.")

tickers = [t.strip() for t in tickers_input.split(",")]
alpha = 1.0 - (confidence_level / 100.0)

# Normalize weights
total_weight = sum(raw_weights)
if total_weight > 0:
    weights = [w / total_weight for w in raw_weights]
else:
    weights = [1.0 / len(tickers)] * len(tickers)

# --- Main Execution (Runs automatically on load & updates on submit) ---
with st.spinner("Fetching market data and calculating risk metrics..."):
    
    # 1. Fetch Data
    prices = fetch_commodity_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    returns = calculate_log_returns(prices)
    
    st.subheader("Next-Day Risk Forecasts")
    
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
    
    normalized_prices = (prices / prices.iloc[0]) * 100
    fig = px.line(normalized_prices, title="Asset Performance (Base 100)", labels={'value': 'Normalized Price', 'Date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    # --- Portfolio Correlation & Covariance ---
    st.markdown("---")
    st.subheader("Diversification Benefit (Covariance Matrix)")
    st.write("This matrix shows how the assets interact. Negative numbers mean they move in opposite directions, acting as a natural hedge.")
    
    cov_matrix = returns[tickers].cov() * 252
    st.dataframe(cov_matrix.style.background_gradient(cmap='coolwarm', axis=None), use_container_width=True)

    # --- Total Portfolio Risk ---
    st.markdown("---")
    st.subheader("Total Portfolio Risk Forecast")
    
    portfolio_returns = (returns[tickers] * weights).sum(axis=1)
    
    with st.spinner("Calculating total portfolio volatility..."):
        port_vol = forecast_garch_volatility(portfolio_returns)
        port_risk_metrics = calculate_conditional_var(portfolio_returns, port_vol, alpha=alpha)
        
        p_vol_pct = port_vol * 100
        p_var_pct = port_risk_metrics['VaR_99'] * 100
        p_cvar_pct = port_risk_metrics['CVaR_99'] * 100
        
        p1, p2, p3 = st.columns(3)
        p1.metric("Total Portfolio Volatility", f"{p_vol_pct:.2f}%")
        p2.metric(f"Portfolio VaR ({confidence_level}%)", f"{p_var_pct:.2f}%")
        p3.metric(f"Portfolio CVaR ({confidence_level}%)", f"{p_cvar_pct:.2f}%")

    # --- Backtesting & Validation UI ---
    if run_backtest:
        st.markdown("---")
        st.subheader("Model Validation: Kupiec Proportion of Failures (POF)")
        
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