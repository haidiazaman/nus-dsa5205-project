import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.graph_objects as go
from tqdm import tqdm
from rf_rate import calculate_mean_rf_rate
import warnings

warnings.filterwarnings("ignore")

stocks = ["DDOG", "SNPS", "BKNG", "SMCI", "MDB", "NVDA", "MELI", "WDAY", "MU", "PDD"]

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 8, 1)
lookback_period = 30
rebalance_interval = 10
rf_rate = calculate_mean_rf_rate(start_date, end_date)

def fetch_stock_data(stocks, start_date, end_date, lookback_period):
    data_start = start_date - timedelta(days=lookback_period)
    data = yf.download(stocks + ['^GSPC'], start=data_start, end=end_date)['Adj Close']
    return data

def portfolio_stats(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([portfolio_return, portfolio_vol, portfolio_return / portfolio_vol])

def neg_sharpe_ratio(weights, returns):
    portfolio_return, portfolio_vol, sharpe = portfolio_stats(weights, returns)
    return -(portfolio_return - rf_rate) / portfolio_vol

def optimize_portfolio_markowitz(returns):
    num_assets = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def run_markowitz_strategy(data, lookback_period, rebalance_interval, start_date):
    results = pd.DataFrame(index=data.loc[start_date:].index, columns=['Returns'] + list(data.columns))
    last_weights = None
    
    for i in tqdm(range(len(results))):
        current_date = results.index[i]
        
        if i == 0 or i % rebalance_interval == 0:
            lookback_start = current_date - timedelta(days=lookback_period)
            lookback_data = data.loc[lookback_start:current_date]
            returns = lookback_data.pct_change().dropna()
            
            if returns.isnull().values.any() or np.isinf(returns).values.any():
                print(f"NaN or infinite values found in returns at date {current_date}")
                if last_weights is None:
                    continue
                weights = last_weights
            else:
                weights = optimize_portfolio_markowitz(returns)
            
            last_weights = weights
        else:
            weights = last_weights
        
        results.loc[current_date, data.columns] = weights
        
        if i < len(results) - 1:
            next_date = results.index[i+1]
            period_return = data.loc[next_date] / data.loc[current_date] - 1
            results.loc[current_date, 'Returns'] = np.dot(weights, period_return)
    
    return results

# Fetch data
data = fetch_stock_data(stocks, start_date, end_date, lookback_period)
data = data.fillna(method='ffill')

print("Running Markowitz strategy...")
results_markowitz = run_markowitz_strategy(data[stocks], lookback_period, rebalance_interval, start_date)

print("Calculating equal-weighted portfolio...")
equal_weights = np.array([1/len(stocks)] * len(stocks))
results_equal = pd.DataFrame(index=data.loc[start_date:].index, columns=['Returns'] + stocks)
results_equal[stocks] = equal_weights

for i in range(len(results_equal)):
    if i < len(results_equal) - 1:
        current_date = results_equal.index[i]
        next_date = results_equal.index[i+1]
        period_return = data[stocks].loc[next_date] / data[stocks].loc[current_date] - 1
        results_equal.loc[current_date, 'Returns'] = np.dot(equal_weights, period_return)

sp500_returns = data['^GSPC'].loc[start_date:].pct_change().fillna(0)
cum_returns_markowitz = (1 + results_markowitz['Returns'].fillna(0)).cumprod()
cum_returns_equal = (1 + results_equal['Returns'].fillna(0)).cumprod()
cum_returns_sp500 = (1 + sp500_returns).cumprod()

def plot_cumulative_returns(cum_returns_markowitz, cum_returns_equal, cum_returns_sp500):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_returns_markowitz.index, y=cum_returns_markowitz, mode='lines', name='Markowitz'))
    fig.add_trace(go.Scatter(x=cum_returns_equal.index, y=cum_returns_equal, mode='lines', name='Equal-Weighted'))
    fig.add_trace(go.Scatter(x=cum_returns_sp500.index, y=cum_returns_sp500, mode='lines', name='S&P 500'))
    
    fig.update_layout(title='Cumulative Returns Comparison',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Returns',
                      legend_title='Strategy')
    
    fig.show()
    fig.write_html("strategies/cumulative_returns_comparison.html")

plot_cumulative_returns(cum_returns_markowitz, cum_returns_equal, cum_returns_sp500)

print("\nFinal Cumulative Returns:")
print(f"Markowitz: {cum_returns_markowitz.iloc[-1]:.4f}")
print(f"Equal-Weighted: {cum_returns_equal.iloc[-1]:.4f}")
print(f"S&P 500: {cum_returns_sp500.iloc[-1]:.4f}")

rf_rate = calculate_mean_rf_rate(start_date, end_date) / 252  # daily risk-free rate
sharpe_markowitz = (results_markowitz['Returns'].mean() - rf_rate) / results_markowitz['Returns'].std() * np.sqrt(252)
sharpe_equal = (results_equal['Returns'].mean() - rf_rate) / results_equal['Returns'].std() * np.sqrt(252)
sharpe_sp500 = (sp500_returns.mean() - rf_rate) / sp500_returns.std() * np.sqrt(252)

print("\nSharpe Ratios:")
print(f"Markowitz: {sharpe_markowitz:.4f}")
print(f"Equal-Weighted: {sharpe_equal:.4f}")
print(f"S&P 500: {sharpe_sp500:.4f}")
print("\nData range for each stock:")
for column in data.columns:
    print(f"{column}: {data[column].first_valid_index()} to {data[column].last_valid_index()}")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_markowitz_multiple_intervals(data, lookback_period, rebalance_intervals, start_date):
    results = {}
    for interval in rebalance_intervals:
        print(f"Running Markowitz strategy with rebalance interval: {interval}")
        result = run_markowitz_strategy(data[stocks], lookback_period, interval, start_date)
        results[f'Interval: {interval}'] = result
    return results

def plot_markowitz_rebalancing_comparison(results):
    fig = go.Figure()
    
    for label, result in results.items():
        cum_returns = (1 + result['Returns'].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, mode='lines', name=label))
    
    fig.update_layout(title='Markowitz Strategy: Rebalancing Interval Comparison',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Returns',
                      legend_title='Rebalancing Interval')
    
    fig.show()
    fig.write_html("strategies/markowitz_rebalancing_comparison.html")

rebalance_intervals = [1, 10, 30]
markowitz_results = run_markowitz_multiple_intervals(data, lookback_period, rebalance_intervals, start_date)
plot_markowitz_rebalancing_comparison(markowitz_results)

print("\nFinal Cumulative Returns for Different Rebalancing Intervals:")
for label, result in markowitz_results.items():
    cum_returns = (1 + result['Returns'].fillna(0)).cumprod()
    print(f"{label}: {cum_returns.iloc[-1]:.4f}")

print("\nSharpe Ratios for Different Rebalancing Intervals:")
for label, result in markowitz_results.items():
    sharpe = (result['Returns'].mean() - rf_rate) / result['Returns'].std() * np.sqrt(252)
    print(f"{label}: {sharpe:.4f}")