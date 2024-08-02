import pandas as pd
import numpy as np
import yfinance as yf
from pytz import timezone
import argparse

# Change stocks and dates here
# stocks = ['NVDA']
start_date = '2024-07-01'
end_date = '2024-07-31'   


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
    port_len = len(portfolio.columns)
    portfolio_values = pd.Series(index=portfolio.index, dtype=float)
    pnl = pd.Series(index=portfolio.index, dtype=float)
    positions = [[] for _ in range(len(portfolio))]
    # $100 investment
    current_value = 100.0
    
    for i in range(len(portfolio)):
        weights = list(portfolio.iloc[i])
        current_prices = list(prices.iloc[i])
        current_value = sum([a*b for a,b in zip(positions[i-1][1],prices.iloc[i])]) \
                + sum([a*b for a,b in zip(positions[i-1][0],prices.iloc[i])]) if i!=0 else current_value
        long_position = [x * current_value / p if x>0 else 0 for x,p in zip(weights, current_prices)] 
        short_position = [-x * current_value / p if x<0 else 0 for x,p in zip(weights, current_prices)] 
        positions[i] = [long_position, short_position]
        if i == 0:
            # Initial investment
            portfolio_values[i] = current_value 
            pnl[i] = 0
        else:
            # value of long/short position
            long_value = sum([a*b for a,b in zip(long_position, prices.iloc[i])])
            short_value = sum([a*b for a,b in zip(short_position, prices.iloc[i])])
            # Calculate profit/loss
            portfolio_values[i] = long_value + short_value
            pnl[i] = portfolio_values[i] - portfolio_values[i-1]
    return portfolio_values, pnl


def baseline(prices):
    # start with $100
    quantity = 10 * np.ones(len(prices.columns)) / prices.iloc[0]
    final = sum(quantity * prices.iloc[-1])
    return (final - 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str)
    args = vars(parser.parse_args())
    portfolio = pd.read_csv(args['file'])
    portfolio['Date'] = pd.to_datetime(portfolio['Date'])
    portfolio = portfolio.set_index(['Date'])
    print(portfolio.head())
    stocks = list(portfolio.columns)
    prices = fetch_price_data(stocks, portfolio.index)
    portfolio_values, pnl = calculate_portfolio_value_and_pnl(portfolio, prices)
    cumulative_pnl = pnl.cumsum()

    print("\nPortfolio Weights ($):")
    print(portfolio.head())
    print("\nFetched Prices (first 5 rows):")
    print(prices)
    print("\nPortfolio Values (first 5 rows):")
    print(portfolio_values)
    print("\nHourly P/L (first 5 rows):")
    print(pnl.head())
    print("\nCumulative P/L (first 5 rows):")
    print(cumulative_pnl.head())
    print(f"\nTotal P/L: ${cumulative_pnl.iloc[-1]:.2f}")
    print(f"Baseline P/L: ${baseline(prices):.2f}")

main()