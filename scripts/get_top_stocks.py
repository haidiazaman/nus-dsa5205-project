import yfinance as yf
import pandas as pd
import numpy as np
import requests

from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def get_nasdaq_symbols():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    symbols = [row.find_all('td')[1].text for row in table.find_all('tr')[1:]]
    return symbols

def get_ftse_symbols():
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    symbols = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[1].text.strip()
        if ticker.endswith('.'):
            symbols.append(ticker + 'L')
        elif ticker.endswith('ICG'):
            pass
        else:
            symbols.append(ticker + '.L')
    return symbols

def get_sgx_symbols():
    return ['C6L.SI', 'D05.SI', 'O39.SI', 'Z74.SI', 'U11.SI', 'G13.SI', 'F34.SI', 'Y92.SI', 'S68.SI', 'A17U.SI']

def get_stock_data(ticker, end_date, days_back=60, interval="60m"):
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=days_back)
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_metrics(df, market='NASDAQ'):
    trading_days_per_year = 252
    if market == 'NASDAQ':
        hours_per_day = 6.5  # NASDAQ trading hours: 9:30 AM to 4:00 PM 
    elif market == 'FTSE':
        hours_per_day = 8.5  # FTSE trading hours: 8:00 AM to 4:30 PM 
    elif market == 'SGX':
        hours_per_day = 8    # SGX trading hours: 9:00 AM to 5:00 PM 

    # log return for each period (hourly)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # total return from the beginning of the period
    df['Cumulative_Return'] = (df['Close'] / df['Close'].iloc[0]) - 1
    # annualized volatility using a 5-day rolling window
    df['Volatility'] = df['Log_Return'].rolling(window=int(hours_per_day * 5)).std() * np.sqrt(hours_per_day * trading_days_per_year)
    # annualized Sharpe ratio
    df['Sharpe'] = (df['Log_Return'].mean() * (hours_per_day * trading_days_per_year)) / (df['Log_Return'].std() * np.sqrt(hours_per_day * trading_days_per_year))
    # 5-day moving average of trading volume
    df['Volume_MA'] = df['Volume'].rolling(window=int(hours_per_day * 5)).mean()
    # 5-day price momentum
    df['Momentum'] = df['Close'] / df['Close'].shift(int(hours_per_day * 5)) - 1
    return df

def score_stock(ticker, market, end_date, days_back=60, allow_short=False):
    df = get_stock_data(ticker, end_date, days_back)
    if df is None or df.empty or len(df) < int(6.5 * 5):
        return None
    
    try:
        df = calculate_metrics(df, market)
        volatility_score = df['Volatility'].mean()
        liquidity_score = (df['Volume'] / df['Volume_MA']).mean()
        momentum_score = df['Momentum'].mean() if allow_short else max(df['Momentum'].mean(), 0)
        sharpe_score = df['Sharpe'].mean() if allow_short else max(df['Sharpe'].mean(), 0)
        return_score = df['Cumulative_Return'].iloc[-1] 
        
        # please play around with the scores below
        if allow_short:
            total_score = (volatility_score * 0.4 +    # higher vol as it provides profit opp in both directions
                        liquidity_score * 0.1 +        # liquid positions to enter and exit
                        np.abs(momentum_score) * 0.1 +
                        np.abs(return_score) * 0.3)     # abs(return) to see if the market moves

        else:
            total_score = (return_score * 0.3 +         # more weight to higher returns
                        volatility_score * 0.35 + 
                        liquidity_score * 0.15 +        # liquid positions to enter and exit
                        momentum_score * 0.1 +
                        sharpe_score * 0.1)             # risk adjusted returns
        return total_score
    
    except Exception as e:
        print(f"Error calculating score for {ticker}: {e}")
        return None

def get_top_stocks(tickers, markets, end_date, days_back=60, n=30, allow_short=False):
    scores = {}
    for ticker, market in zip(tickers, markets):
        score = score_stock(ticker, market, end_date, days_back, allow_short)
        if score is not None:
            scores[ticker] = score
    
    top_stocks = sorted(scores, key=scores.get, reverse=True)[:n]
    return top_stocks

def generate_top_stocks_df():
    nasdaq_symbols = get_nasdaq_symbols()
    ftse_symbols = get_ftse_symbols()
    sgx_symbols = get_sgx_symbols()
    all_symbols = nasdaq_symbols + ftse_symbols + sgx_symbols
    all_markets = ['NASDAQ'] * len(nasdaq_symbols) + ['FTSE'] * len(ftse_symbols) + ['SGX'] * len(sgx_symbols)
    
    end_date = "2023-12-31"
    days_back = 360
    
    top_stocks_long = get_top_stocks(all_symbols, all_markets, end_date, days_back, n=30, allow_short=False)
    top_stocks_long_short = get_top_stocks(all_symbols, all_markets, end_date, days_back, n=30, allow_short=True)
    
    # print("Top 30 stocks (Long-Only Strategy):")
    # for i, stock in enumerate(top_stocks_long, 1):
    #     print(f"{i}. {stock}")
    
    # print("\nTop 30 stocks (Long-Short Strategy):")
    # for i, stock in enumerate(top_stocks_long_short, 1):
    #     print(f"{i}. {stock}")
    
    pd.DataFrame({'Rank': range(1, 31), 'Ticker': top_stocks_long}).to_csv('stock-selection/top_30_stocks_long_only.csv', index=False)
    pd.DataFrame({'Rank': range(1, 31), 'Ticker': top_stocks_long_short}).to_csv('stock-selection/top_30_stocks_long_short.csv', index=False)
    
    print("\nResults saved to 'top_30_stocks_long_only.csv' and 'top_30_stocks_long_short.csv'")
    
    if top_stocks_long:
        top_stock = top_stocks_long[0]
        df = get_stock_data(top_stock, end_date, days_back)
        if df is not None:
            df = calculate_metrics(df)
            print(f"\nDetailed metrics for top stock (Long-Only Strategy) {top_stock}:")
            print(f"Final Close Price: {df['Close'].iloc[-1]:.2f}")
            print(f"60-day Return: {df['Cumulative_Return'].iloc[-1] * 100:.2f}%")
            print(f"Average Hourly Volatility: {df['Volatility'].mean():.4f}")
            print(f"Average Sharpe Ratio: {df['Sharpe'].mean():.4f}")
            print(f"Average Hourly Volume: {df['Volume'].mean():.0f}")
    
    return top_stocks_long, top_stocks_long_short

def get_common_top_stocks(top_stocks_long, top_stocks_long_short):
    common_top_stocks = set(top_stocks_long).union(set(top_stocks_long_short))
    print()
    print("num common_top_stocks: ",len(common_top_stocks))
    # print("common_top_stocks: ",common_top_stocks)
    
    return common_top_stocks
    
    

if __name__ == "__main__":
    top_stocks_long, top_stocks_long_short = generate_top_stocks_df()
    common_top_stocks = get_common_top_stocks(top_stocks_long, top_stocks_long_short)