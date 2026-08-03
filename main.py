from data.market_data import fetch_commodity_data, calculate_log_returns

if __name__ == "__main__":
    portfolio_tickers = ['CL=F', 'GC=F', 'SI=F']
    
    prices = fetch_commodity_data(portfolio_tickers, start_date="2021-01-01", end_date="2026-08-01")
    
    # Transform to log returns
    returns = calculate_log_returns(prices)
    
    print("\n--- Daily Log Returns ---")
    print(returns.tail())