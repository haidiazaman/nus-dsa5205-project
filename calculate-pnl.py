import pandas as pd
import numpy as np
import yfinance as yf
from pytz import timezone

# Change stocks and dates here
stocks = ['NVDA', 'SMCI', 'DASH'] 
start_date = '2023-01-03'
end_date = '2023-06-30'   

def generate_portfolio_data(stocks, start_date, end_date):
    """
    Dummy portfolio - Bryan this is supposed to simulate your data:

    Stimulated  Portfolio Weights ($):
                                NVDA      SMCI      DASH
    2023-01-03 09:30:00-05:00  0.345965  0.316687  0.337348
    2023-01-03 10:30:00-05:00  0.319118  0.331961  0.348920
    2023-01-03 11:30:00-05:00  0.343183  0.323547  0.333270
    2023-01-03 12:30:00-05:00  0.339176  0.346195  0.314629
    2023-01-03 13:30:00-05:00  0.334206  0.334142  0.331652
    """

    ny_tz = timezone('America/New_York')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_hours = pd.date_range(start='09:30', end='16:00', freq='1h').time
    
    dates = []
    for date in date_range:
        for time in trading_hours:
            dates.append(ny_tz.localize(pd.Timestamp.combine(date, time)))
    
    num_time_points = len(dates)
    num_stocks = len(stocks)
    base_weights = np.full((num_time_points, num_stocks), 1 / num_stocks)
    noise = np.random.normal(0, 0.01, (num_time_points, num_stocks))
    noise -= noise.mean(axis=1, keepdims=True)
    weights = base_weights + noise
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    df = pd.DataFrame(
        weights,
        columns=stocks,
        index=dates
    )
    df = df[df.index.dayofweek < 5]
    
    return df

def fetch_price_data(stocks, portfolio_dates):
    price_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        start_date = portfolio_dates.min().strftime('%Y-%m-%d')
        end_date = (portfolio_dates.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch 1-hour interval data for the entire date range
        data = ticker.history(start=start_date, end=end_date, interval='1h')
        data.index = data.index.tz_convert('America/New_York')
        
        # Reindex to match portfolio dates, using nearest available price
        aligned_data = data['Close'].reindex(portfolio_dates, method='nearest')
        price_data[stock] = aligned_data
    
    df = pd.DataFrame(price_data)
    return df

def calculate_portfolio_value_and_pnl(portfolio, prices):
    portfolio_values = pd.Series(index=portfolio.index, dtype=float)
    pnl = pd.Series(index=portfolio.index, dtype=float)
    
    # $1 investment
    current_value = 1.0
    
    for i in range(len(portfolio)):
        if i == 0:
            # Initial investment
            portfolio_values[i] = current_value
            pnl[i] = 0
        else:
            # Sell previous portfolio
            current_value = sum(previous_shares * prices.iloc[i])
            
            # Calculate profit/loss
            pnl[i] = current_value - portfolio_values[i-1]
            
            # Record new portfolio value
            portfolio_values[i] = current_value
        
        # Buy new portfolio
        weights = portfolio.iloc[i]
        current_prices = prices.iloc[i]
        previous_shares = weights * current_value / current_prices
    
    return portfolio_values, pnl

portfolio = generate_portfolio_data(stocks, start_date, end_date)
prices = fetch_price_data(stocks, portfolio.index)
portfolio_values, pnl = calculate_portfolio_value_and_pnl(portfolio, prices)
cumulative_pnl = pnl.cumsum()

print("\nPortfolio Weights ($):")
print(portfolio.head())
print("\nFetched Prices (first 5 rows):")
print(prices.head())
print("\nPortfolio Values (first 5 rows):")
print(portfolio_values.head())
print("\nHourly P/L (first 5 rows):")
print(pnl.head())
print("\nCumulative P/L (first 5 rows):")
print(cumulative_pnl.head())
print(f"\nTotal P/L from {start_date} to {end_date}: ${cumulative_pnl.iloc[-1]:.2f}")