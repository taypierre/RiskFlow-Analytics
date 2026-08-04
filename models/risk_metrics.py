import numpy as np
import pandas as pd

def calculate_conditional_var(historical_returns: pd.Series, forecasted_vol: float, alpha: float = 0.01) -> dict:
    """
    Calculates the Value at Risk (VaR) and Expected Shortfall (CVaR) 
    scaled to the GARCH volatility forecast.
    
    alpha=0.01 represents a 99% confidence interval.
    """
    # Find the baseline historical volatility
    historical_vol = historical_returns.std()
    
    # 2. Scale the historical returns to match tomorrow's GARCH forecast
    # If GARCH predicts high volatility, this magnifies past losses.
    scaled_returns = historical_returns * (forecasted_vol / historical_vol)
    
    # 3. Calculate VaR: The worst loss at the 99th percentile
    var_limit = np.percentile(scaled_returns, alpha * 100)
    
    # 4. Calculate CVaR: The average of all losses worse than the VaR limit
    tail_losses = scaled_returns[scaled_returns <= var_limit]
    cvar = tail_losses.mean()
    
    return {
        "VaR_99": var_limit,
        "CVaR_99": cvar
    }