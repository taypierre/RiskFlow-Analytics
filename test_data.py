import yfinance as yf

# 1. Define the test ticker
test_ticker = "AAPL"

print(f"Fetching data for {test_ticker}...")

# 2. Download the data (matching the logic in your app.py)
data = yf.download(test_ticker, start="2021-01-01", end="2026-08-10")['Close']

# 3. Print the results to verify it pulled correctly
print("\n--- FIRST 5 DAYS ---")
print(data.head())

print("\n--- LAST 5 DAYS ---")
print(data.tail())

print(f"\nTotal data points pulled: {len(data)}")