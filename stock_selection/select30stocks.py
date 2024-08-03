import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from trading_days import calculate_trading_days
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def get_stock_universe():
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
        url = "https://www.tradingview.com/markets/stocks-singapore/market-movers-large-cap/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr', attrs={'data-rowkey': True})
        symbols = [row['data-rowkey'].replace('SGX:', '') + '.SI' for row in rows[:20] if row['data-rowkey'] != 'SGX:5E2']
        return symbols

    return get_nasdaq_symbols() + get_ftse_symbols() + get_sgx_symbols()

def get_stock_data(ticker, end_date, days_back, interval="60m"):
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

def calculate_metrics(df, market='NASDAQ', start_date=None, end_date=None):
    days = calculate_trading_days(start_date, end_date, market)
    if market == 'NASDAQ':
        hours = 6.5     # NASDAQ trading hours: 9:30 AM to 4:00 PM 
    elif market == 'FTSE':
        hours = 8.5     # FTSE trading hours: 8:00 AM to 4:30 PM 
    elif market == 'SGX':
        hours = 8       # SGX trading hours: 9:00 AM to 5:00 PM 
        
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))       # continuous compounded return between two consecutive periods
    df['Cumulative_Return'] = (df['Close'] / df['Close'].iloc[0]) - 1   # total percentage return from the starting point

    for window in [30, 60, 90]:
        df[f'Momentum_{window}d'] = df['Close'] / df['Close'].shift(int(hours * window)) - 1    # rate of acceleration of a stock's price over {window} days
        df[f'Volume_EMA_{window}d'] = df['Volume'].ewm(span=int(hours * window)).mean()         # smoothed average of trading volume over {window} days
        df[f'Sharpe_{window}d'] = (df['Log_Return'].rolling(window=int(hours * window)).mean() * (hours * days)) / (df['Log_Return'].rolling(window=int(hours * window)).std() * np.sqrt(hours * days)) # risk-adjusted return, calculated as the annualized return divided by annualized volatility
        df[f'Volatility_{window}d'] = df['Log_Return'].rolling(window=int(hours * window)).std() * np.sqrt(hours * days)    # stock's price fluctuations over {window} days, annualized
    
    return df

def prepare_data_for_ml(universe, markets, end_date, days_back):
    data = []
    start_date = end_date - pd.Timedelta(days=days_back)
    for ticker, market in zip(universe, markets):
        days = calculate_trading_days(start_date, end_date, market)
        df = get_stock_data(ticker, end_date, days_back)

        if df is not None and not df.empty:
            df = calculate_metrics(df, market, start_date, end_date)
            stock_data = {
                'ticker': ticker,
                'market': market,
                'return': df['Cumulative_Return'].iloc[-1],
                'future_return': df['Log_Return'].shift(-1).mean() * days
            }
            for window in [30, 60, 90]:
                stock_data.update({
                    f'momentum_{window}d': df[f'Momentum_{window}d'].mean(),
                    f'liquidity_{window}d': (df['Volume'] / df[f'Volume_EMA_{window}d']).mean(),
                    f'sharpe_{window}d': df[f'Sharpe_{window}d'].mean(),
                    f'volatility_{window}d': df[f'Volatility_{window}d'].mean(),
                })
            data.append(stock_data)

        scaler = StandardScaler()
        numerical_columns = [col for col in df.columns if col not in ['ticker', 'market', 'future_return']]
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return pd.DataFrame(data)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_random_forest(data):
    le = LabelEncoder()
    data['market_encoded'] = le.fit_transform(data['market'])
    
    numerical_columns = ['market_encoded'] + [col for col in data.columns if col.startswith(('momentum_', 'liquidity_', 'sharpe_', 'volatility_'))]
    X = data[numerical_columns]
    y = data['future_return']

    split_point = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Performance measurement
    train_predictions = rf.predict(X_train)
    test_predictions = rf.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print("Random Forest Performance Metrics:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Training R-squared: {train_r2:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    
    return rf, le, numerical_columns, (test_rmse, test_mae, test_r2)


def rank_stocks_rf(rf_model, data, le):
    data['market_encoded'] = le.transform(data['market'])
    numerical_columns = ['market_encoded'] + [col for col in data.columns if col.startswith(('momentum_', 'liquidity_', 'sharpe_', 'volatility_'))]
    X = data[numerical_columns]
    predictions = rf_model.predict(X)
    data['predicted_return'] = predictions
    return data.sort_values('predicted_return', ascending=False)['ticker'].tolist()

end_date = "2023-12-31"
days_back = 365

universe = get_stock_universe()
markets  = ['NASDAQ'] * 100 + ['FTSE'] * 100 + ['SGX'] * 10
end_date = datetime.strptime(end_date, "%Y-%m-%d")
start_date = end_date - timedelta(days = days_back)  

data = prepare_data_for_ml(universe, markets, end_date, days_back)
rf_model, le, feature_names, (test_rmse, test_mae, test_r2) = train_random_forest(data)
top_30_stocks = [stock for stock in rank_stocks_rf(rf_model, data, le)[:31] if stock not in ('ARM')]

print("Top 30 stocks:")
print(top_30_stocks)
pd.Series(top_30_stocks, name='Ticker').to_csv('stock_selection/top_30_stocks.csv', index=False)
print("Top 30 stocks saved to 'top_30_stocks.csv'")

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)
feature_importance.to_csv('feature_importance.csv', index=False)

