import numpy as np
import pandas as pd
import scipy.stats as stats
from models.volatility import forecast_garch_volatility
from models.risk_metrics import calculate_conditional_var

def run_historical_backtest(returns, window_size=252, alpha=0.01):
    """
    Runs a rolling-window backtest for VaR.
    """
    breaches = 0
    total_days = len(returns) - window_size
    
    results = []

    for i in range(total_days):
        # Isolate the historical window
        historical_window = returns.iloc[i : i + window_size]
        
        # Get the actual return for the next day after the window
        actual_next_day_return = returns.iloc[i + window_size]
        
        # Forecast risk for that next day
        try:
            pred_vol = forecast_garch_volatility(historical_window)
            risk_metrics = calculate_conditional_var(historical_window, pred_vol, alpha=alpha)
            
            var_prediction = -abs(risk_metrics['VaR_99']) 
            
            # Check if the actual return breaches the predicted VaR
            is_breach = actual_next_day_return < var_prediction
            if is_breach:
                breaches += 1
                
            results.append({
                'Date': returns.index[i + window_size],
                'Actual_Return': actual_next_day_return,
                'Predicted_VaR': var_prediction,
                'Breach': is_breach
            })
            
        except Exception as e:
            continue

    # Calculate the breach rate
    actual_breach_rate = breaches / total_days
    
    expected_breach_rate = alpha 
    
    return pd.DataFrame(results), actual_breach_rate, expected_breach_rate, breaches, total_days

def kupiec_pof_test(expected_rate, actual_breaches, total_days, confidence_level=0.95):
    """
    Performs the Kupiec POF Test to validate a VaR model.
    Returns a Pass/Fail boolean, the test statistic (LR), and the critical value.
    """
    p = expected_rate
    x = actual_breaches
    N = total_days
    p_hat = x / N

    if x == 0:
        lr = -2 * (N * np.log(1 - p))
    elif x == N:
        lr = -2 * (N * np.log(p))
    else:
        ll_null = (N - x) * np.log(1 - p) + x * np.log(p)
        ll_alt = (N - x) * np.log(1 - p_hat) + x * np.log(p_hat)
        lr = -2 * (ll_null - ll_alt)

    # Calculate the Chi-Square critical value (1 degree of freedom)
    critical_value = stats.chi2.ppf(confidence_level, 1)
    
    # Passes if the Likelihood Ratio is less than the critical value
    model_passed = lr < critical_value
    
    return model_passed, lr, critical_value