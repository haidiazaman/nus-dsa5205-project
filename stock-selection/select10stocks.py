import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

top_30_stocks = pd.read_csv('stock-selection/top_30_stocks_long_only.csv')['Ticker'].tolist()

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Get stock data
start_date = '2023-01-01'
end_date = '2023-12-31'
stock_data = get_stock_data(top_30_stocks, start_date, end_date)

# Calculate returns
returns = stock_data.pct_change().dropna()

# Correlation analysis
correlation_matrix = returns.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Top 30 Stocks')
plt.savefig('stock-selection/correlation_heatmap.png')
plt.close()

# Mean-Variance Optimization
# Calculate expected returns and covariance
expected_returns = returns.mean() * 252  # Annualized returns
cov_matrix = returns.cov() * 252  # Annualized covariance

# Define portfolio optimization functions
def portfolio_return(weights, expected_returns):
    return np.sum(expected_returns * weights)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.04):
    p_return = portfolio_return(weights, expected_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

# Optimization constraints
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(len(top_30_stocks)))

# Perform optimization
initial_weights = np.array([1/len(top_30_stocks)] * len(top_30_stocks))
optimized = minimize(neg_sharpe_ratio, initial_weights, 
                     args=(expected_returns, cov_matrix),
                     method='SLSQP', bounds=bounds, constraints=constraints)

# Get the optimized weights
optimized_weights = optimized.x

# Select top 10 stocks based on optimized weights
top_10_indices = np.argsort(optimized_weights)[-10:]
top_10_stocks = [top_30_stocks[i] for i in top_10_indices]
top_10_weights = optimized_weights[top_10_indices]

# Normalize weights for the top 10 stocks
top_10_weights = top_10_weights / np.sum(top_10_weights)

# Print results
print("Top 10 stocks selected:")
for stock, weight in zip(top_10_stocks, top_10_weights):
    print(f"{stock}: {weight:.4f}")

# Calculate portfolio metrics
portfolio_return = portfolio_return(top_10_weights, expected_returns[top_10_indices])
portfolio_volatility = portfolio_volatility(top_10_weights, cov_matrix.iloc[top_10_indices, top_10_indices])
portfolio_sharpe = (portfolio_return - 0.02) / portfolio_volatility

print(f"\nPortfolio Metrics:")
print(f"Expected Annual Return: {portfolio_return:.4f}")
print(f"Annual Volatility: {portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {portfolio_sharpe:.4f}")

# Save results
results = pd.DataFrame({
    'Stock': top_10_stocks,
    'Weight': top_10_weights
})
results.to_csv('stock-selection/chosen_10_stocks.csv', index=False)
print("\nOptimized portfolio saved to 'chosen_10_stocks.csv'")