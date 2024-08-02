from polygon import RESTClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

polygon_client = RESTClient('r3ga8e4I8fHrnv1iZNHpikJZk2VjWrmv')

def get_intraday_data(symbol, start_date, end_date):
    aggs = polygon_client.get_aggs(symbol, 5, 'minute', start_date, end_date)
    df = pd.DataFrame(aggs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_relative_volume(data, lookback=14):
    daily_volume = data.resample('D')['volume'].sum()
    return daily_volume / daily_volume.rolling(window=lookback).mean()

def orb_strategy(symbol, start_date, end_date):
    data = get_intraday_data(symbol, start_date, end_date)
    
    daily_data = data.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    daily_data['RelativeVolume'] = calculate_relative_volume(daily_data)
    
    opening_range = data.groupby(data.index.date).first()
    opening_range.index = pd.to_datetime(opening_range.index)
    

    common_index = daily_data.index.intersection(opening_range.index)
    daily_data = daily_data.loc[common_index]
    opening_range = opening_range.loc[common_index]
    
    daily_data['Signal'] = np.where(
        (daily_data['high'] > opening_range['high']) & (daily_data['open'] > daily_data['close'].shift(1)), 1,
        np.where((daily_data['low'] < opening_range['low']) & (daily_data['open'] < daily_data['close'].shift(1)), -1, 0)
    )
    
    daily_data['Returns'] = daily_data['close'].pct_change()
    daily_data['Strategy'] = daily_data['Signal'].shift(1) * daily_data['Returns']
    
    return daily_data

stocks = ['SMCI', 'NVDA', 'META', 'CRWD', 'AMD', 'TSLA', 'MDB', 'PANW', 'AVGO']

start_date = '2023-01-01' 
end_date = '2023-12-31'

results = {}
for symbol in stocks:
    try:
        result = orb_strategy(symbol, start_date, end_date)
        results[symbol] = result['Strategy'].cumsum().iloc[-1]
        print(f"Processed {symbol} successfully")
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")

try:
    sp500 = get_intraday_data('SPY', start_date, end_date)
    sp500_daily = sp500.resample('D').last()
    sp500_return = sp500_daily['close'].pct_change().cumsum().iloc[-1]
    print("Processed S&P 500 successfully")
except Exception as e:
    print(f"Error processing S&P 500: {str(e)}")
    sp500_return = None

for symbol, returns in results.items():
    print(f"{symbol} Cumulative Return: {returns:.2f}")
if sp500_return is not None:
    print(f"S&P 500 Cumulative Return: {sp500_return:.2f}")
else:
    print("S&P 500 data not available")