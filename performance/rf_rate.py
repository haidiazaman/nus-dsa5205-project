import pandas as pd
import numpy as np
import yfinance as yf

def calculate_mean_rf_rate(start_date, end_date):
    # Fetch ^IRX data (13-week Treasury Bill), equivalent to 3-month treasury bill
    irx = yf.Ticker("^IRX")
    hist = irx.history(start=start_date, end=end_date)
    mean_rf_rate = hist['Close'].mean() / 100 
    
    return mean_rf_rate

# Example usage
start_date = '2024-01-03'
end_date = '2024-06-30'

mean_rf = calculate_mean_rf_rate(start_date, end_date)
print(f"Mean risk-free rate between {start_date} and {end_date}: {mean_rf:.4f}")

# Calculate annualized rate
annualized_rf = (1 + mean_rf) ** 4 - 1  # Compounded quarterly
print(f"Annualized risk-free rate: {annualized_rf:.4f}")