import yfinance as yf
import pandas as pd
import numpy as np

def fetch_commodity_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches adjusted closing prices for a list of commodity tickers.
    """
    print(f"Fetching data for {tickers}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Forward fill any missing data points
    data = data.ffill()
    return data

def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw prices into daily log returns.
    """
    log_returns = np.log(prices / prices.shift(1))
    
    # Drop the first row
    return log_returns.dropna()