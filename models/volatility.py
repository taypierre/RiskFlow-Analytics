from arch import arch_model
import pandas as pd
import numpy as np

def forecast_garch_volatility(returns: pd.Series) -> float:
    """
    Fits a GARCH(1,1) model to a series of log returns and 
    forecasts the next day's volatility (standard deviation).
    """
    # Scale returns to percentage
    scaled_returns = returns * 100
    
    # Define the GARCH(1,1) model
    # p=1 (looks at yesterday's variance)
    # q=1 (looks at yesterday's return shock)
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
    
    # disp='off' hides the optimization iterations
    fitted_model = model.fit(disp='off')
    
    # Forecast the variance for the next 1 day
    forecasts = fitted_model.forecast(horizon=1)
    
    next_day_variance = forecasts.variance.iloc[-1, 0]
    
    # Volatility is the square root of variance
    next_day_vol = np.sqrt(next_day_variance) / 100
    
    return next_day_vol