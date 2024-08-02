import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from calculate_pnl import generate_portfolio_data, fetch_price_data, calculate_portfolio_value_and_pnl, stocks, start_date, end_date
from trading_days import calculate_trading_days
from rf_rate import calculate_mean_rf_rate

risk_free_rate = calculate_mean_rf_rate(start_date, end_date)
trading_days = calculate_trading_days(start_date, end_date)
trading_hours = 6.5

def fetch_snp500_data(start_date, end_date, portfolio_index):
    """Fetch S&P 500 data and align it with portfolio dates"""
    snp500 = yf.Ticker("^GSPC")
    snp500_data = snp500.history(start=start_date, end=end_date, interval="1h")
    snp500_data.index = snp500_data.index.tz_convert('America/New_York')
    return snp500_data['Close'].reindex(portfolio_index, method='ffill')

def calculate_ratios(portfolio_values, benchmark_values):
    portfolio_returns = portfolio_values.pct_change().dropna()
    benchmark_returns = benchmark_values.pct_change().dropna()
    
    annualization_factor = np.sqrt(trading_days * trading_hours)
    excess_returns = portfolio_returns - risk_free_rate / (trading_days * trading_hours)
    
    sharpe_ratio = (excess_returns.mean() * annualization_factor) / (excess_returns.std() * annualization_factor)
    active_returns = portfolio_returns - benchmark_returns
    information_ratio = (active_returns.mean() * annualization_factor) / (active_returns.std() * annualization_factor)
    
    return sharpe_ratio, information_ratio

# Generate portfolio data and calculate PNL
portfolio = generate_portfolio_data(stocks, start_date, end_date)
prices = fetch_price_data(stocks, portfolio.index)
portfolio_values, pnl = calculate_portfolio_value_and_pnl(portfolio, prices)
snp500_values = fetch_snp500_data(start_date, end_date, portfolio.index)

# Calculate ratios
sharpe_ratio, information_ratio = calculate_ratios(portfolio_values, snp500_values)

print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Information Ratio: {information_ratio:.4f}")

# Calculate cumulative returns
portfolio_cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
snp500_cumulative_return = (snp500_values.iloc[-1] / snp500_values.iloc[0]) - 1

print(f"Portfolio Cumulative Return: {portfolio_cumulative_return:.2%}")
print(f"S&P 500 Cumulative Return: {snp500_cumulative_return:.2%}")

# Plot portfolio vs S&P 500 performance
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values / portfolio_values.iloc[0], label='Portfolio')
plt.plot(snp500_values / snp500_values.iloc[0], label='S&P 500')
plt.title('Portfolio vs S&P 500 Performance')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()