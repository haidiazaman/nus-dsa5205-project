import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# Download NASDAQ-100 components
nasdaq100 = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
tickers = nasdaq100['Ticker'].tolist()

# Add S&P 500 ETF for comparison
tickers.append('^GSPC')

# Set date ranges
start_date = "2023-12-25"
end_date = "2023-12-31"
evaluation_start = "2024-01-01"
evaluation_end = datetime.now().strftime('%Y-%m-%d')

# Download data for the optimization period
data_opt = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns for the optimization period
returns_opt = data_opt.pct_change().dropna()

# Calculate mean returns and covariance for the optimization period
mean_returns_opt = returns_opt.mean()
cov_matrix_opt = returns_opt.cov()

# Calculate the number of trading days in a year
trading_days = len(returns_opt)
annualization_factor = trading_days / (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * 365.25

# Define portfolio optimization functions
def portfolio_return(weights, mean_returns):
    return np.sum(mean_returns * weights) * annualization_factor

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(annualization_factor)

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    p_ret = portfolio_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

# Optimization
def optimize_portfolio(mean_returns, cov_matrix, num_stocks):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Optimize for all NASDAQ-100 stocks (excluding S&P 500)
result = optimize_portfolio(mean_returns_opt.drop('^GSPC'), cov_matrix_opt.drop('^GSPC').drop('^GSPC', axis=1), 10)

# Get top 10 stocks
top_10_weights = pd.Series(result.x, index=mean_returns_opt.drop('^GSPC').index).nlargest(10)
top_10_stocks = top_10_weights.index.tolist()

print("Top 10 stocks selected based on data from {} to {}:".format(start_date, end_date))
for stock, weight in top_10_weights.items():
    print(f"{stock}: {weight:.4f}")

# Download data for the evaluation period
data_eval = yf.download(top_10_stocks + ['^GSPC'], start=evaluation_start, end=evaluation_end)['Adj Close']

# Calculate daily returns for the evaluation period
returns_eval = data_eval.pct_change().dropna()

# Calculate portfolio returns
portfolio_returns = (returns_eval[top_10_stocks] * top_10_weights).sum(axis=1)

# Calculate cumulative returns
cumulative_returns_portfolio = (1 + portfolio_returns).cumprod()
cumulative_returns_sp500 = (1 + returns_eval['^GSPC']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns_portfolio, label='Optimized Portfolio')
plt.plot(cumulative_returns_sp500, label='S&P 500')
plt.title(f'Cumulative Returns: Optimized Portfolio vs S&P 500 ({evaluation_start} to {evaluation_end})')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Calculate performance metrics
portfolio_total_return = cumulative_returns_portfolio.iloc[-1] - 1
sp500_total_return = cumulative_returns_sp500.iloc[-1] - 1

# Calculate annualized returns and volatility
eval_trading_days = len(returns_eval)
eval_annualization_factor = eval_trading_days / (pd.to_datetime(evaluation_end) - pd.to_datetime(evaluation_start)).days * 365.25

portfolio_annual_return = (1 + portfolio_total_return) ** (eval_annualization_factor/eval_trading_days) - 1
sp500_annual_return = (1 + sp500_total_return) ** (eval_annualization_factor/eval_trading_days) - 1
portfolio_annual_volatility = portfolio_returns.std() * np.sqrt(eval_annualization_factor)
sp500_annual_volatility = returns_eval['^GSPC'].std() * np.sqrt(eval_annualization_factor)

# Calculate Sharpe Ratio (assuming risk-free rate of 2%)
risk_free_rate = 0.02
portfolio_sharpe = (portfolio_annual_return - risk_free_rate) / portfolio_annual_volatility
sp500_sharpe = (sp500_annual_return - risk_free_rate) / sp500_annual_volatility

print(f"\
Performance from {evaluation_start} to {evaluation_end}:")
print(f"Optimized Portfolio Total Return: {portfolio_total_return:.4f}")
print(f"S&P 500 Total Return: {sp500_total_return:.4f}")
print(f"\
Optimized Portfolio Annualized Return: {portfolio_annual_return:.4f}")
print(f"S&P 500 Annualized Return: {sp500_annual_return:.4f}")
print(f"\
Optimized Portfolio Annualized Volatility: {portfolio_annual_volatility:.4f}")
print(f"S&P 500 Annualized Volatility: {sp500_annual_volatility:.4f}")
print(f"\
Optimized Portfolio Sharpe Ratio: {portfolio_sharpe:.4f}")
print(f"S&P 500 Sharpe Ratio: {sp500_sharpe:.4f}")