from data.market_data import fetch_commodity_data, calculate_log_returns
from models.volatility import forecast_garch_volatility
from models.risk_metrics import calculate_conditional_var

if __name__ == "__main__":
    portfolio_tickers = ['CL=F', 'GC=F', 'SI=F']
    
    # Fetch & Transform Data
    prices = fetch_commodity_data(portfolio_tickers, start_date="2021-01-01", end_date="2026-08-01")
    returns = calculate_log_returns(prices)
    
    print("\n--- 99% Risk Metrics (Scaled by GARCH) ---")
    
    for ticker in portfolio_tickers:
        asset_returns = returns[ticker]
        
        # Forecast Volatility
        predicted_vol = forecast_garch_volatility(asset_returns)
        
        # Calculate Risk (Expected Shortfall)
        risk_metrics = calculate_conditional_var(asset_returns, predicted_vol)
        
        # Results
        var_pct = risk_metrics['VaR_99'] * 100
        cvar_pct = risk_metrics['CVaR_99'] * 100
        
        print(f"\n{ticker} Forecasted Volatility: {predicted_vol * 100:.2f}%")
        print(f"{ticker} 99% VaR:  {var_pct:.2f}% (Threshold)")
        print(f"{ticker} 99% CVaR: {cvar_pct:.2f}% (Expected Shortfall)")