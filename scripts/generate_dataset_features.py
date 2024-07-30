import yfinance as yf
import pandas as pd
import numpy as np
import requests
import argparse
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def download_data_from_yf(stock_name,start_date,end_date,interval):
    """
    data download from yahoo finance
    """
    stock_data = yf.download(stock_name,start_date,end_date,interval=interval)
    return stock_data


# functions to generate features
def insert_ema_column(df, hours, column_to_ema, ema_column_name):
    side_df = df.copy()
    multiplier = 2 / (1 + hours)
    side_df.loc[hours, ema_column_name] = side_df.iloc[:hours][column_to_ema].mean()
    for day in range(hours + 1, len(side_df)):
        side_df.loc[day, ema_column_name] = side_df.loc[day - 1, ema_column_name] * (1 - multiplier) + side_df.loc[day, column_to_ema] * multiplier

    side_df[ema_column_name] = (side_df[column_to_ema] / side_df[ema_column_name] - 1).clip(-1, 1)
    return side_df

def insert_sma_column(df, hours, column_to_sma, sma_column_name):
    side_df = df.copy()
    side_df[sma_column_name] = side_df[column_to_sma].rolling(hours).mean()

    side_df[sma_column_name] = (side_df[column_to_sma] / side_df[sma_column_name] - 1).clip(-1, 1)
    
    return side_df

def insert_vwap_column(df, hours, vwap_column_name):
    side_df = df.copy()
    side_df[vwap_column_name] = side_df['Close'] * side_df['Volume']
    side_df[vwap_column_name] = side_df[vwap_column_name].rolling(hours).sum() / side_df['Volume'].rolling(hours).sum()

    side_df[vwap_column_name] = (side_df['Close'] / side_df[vwap_column_name] - 1).clip(-1, 1)
    
    return side_df

def insert_stddev_column(df, hours, stddev_column_name):
    side_df = df.copy()
    side_df[stddev_column_name] = side_df['Close'].rolling(hours).std()
    
    return side_df

def insert_fama_french_column(df):

    ff_factors = pd.read_csv('Data/famafrench_daily_factor.csv')
    ff_factors.drop(['Date_str'], axis=1, inplace=True)
    ff_factors = ff_factors.set_index('Date')
    ff_factors.index = pd.to_datetime(ff_factors.index)
    
    fdf = ff_factors.reset_index()
    fdf = fdf[['Date', 'Mkt-RF']].rename(columns={'Mkt-RF': 'FamaFrenchMktReturns'})
    fdf = fdf[
        fdf['Date'] >= '2023-01-01'
    ]
    fdf['Date'] = pd.to_datetime(fdf['Date'])
    
    side_df = df.copy()
    side_df = side_df.reset_index(drop=True)
    side_df['Date'] = pd.to_datetime(side_df['Datetime'].dt.date)
    side_df = side_df.merge(fdf, how='left', on='Date')
    side_df = side_df.drop(['Date'], axis=1)
    side_df['FamaFrenchMktReturns'] = side_df['FamaFrenchMktReturns'].ffill()
    return side_df



def generate_features(df,trading_days_per_year = 252, hours_per_day = 6.5):
    """
    generate features df for 1 stock
    """
    df = df.reset_index()
    
    # log return for each period (hourly)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    hours_gridsearch = [2 ** i for i in range(1, 9)]
    
    for index, hour in enumerate(hours_gridsearch):
        ema_column_name = f"EMAVolumeDiff{hour}"
        df = insert_ema_column(df, hour, 'Volume', ema_column_name)
    
        sma_column_name = f"SMAVolumeDiff{hour}"
        df = insert_sma_column(df, hour, 'Volume', sma_column_name)
    
        ema_column_name = f"EMACloseDiff{hour}"
        df = insert_ema_column(df, hour, 'Close', ema_column_name)
        
        sma_column_name = f"SMACloseDiff{hour}"
        df = insert_sma_column(df, hour, 'Close', sma_column_name)
    
        vwap_column_name = f"VWAP{hour}"
        df = insert_vwap_column(df, hour, vwap_column_name)
    
        stddev_column_name = f"VolatilityStdDev{hour}"
        df = insert_stddev_column(df, hour, stddev_column_name)
        
        # annualized volatility using a 5-day rolling window
        df[f'Volatility{hour}'] = df['Log_Return'].rolling(window=hour).std() * np.sqrt(hours_per_day * trading_days_per_year)
        
        # momentum from t-x hour
        df[f'Momentum{hour}'] = (df['Close'] / df['Close'].shift(hour)).clip(-1, 1)
    
    df['PriceVolatilityHourly'] = ((df['High'] - df['Low']) - 1).clip(-1, 1)
    
    ### brb fixing normalization
    for index, hour in enumerate(hours_gridsearch):
        if index <= 1:
            continue
    
        longer = f'EMACloseDiff{hour}'
        shorter = f'EMACloseDiff{hours_gridsearch[index - 1]}'
        signal = f'EMACloseDiff{hours_gridsearch[index - 2]}'
        df[f'MACD{hour}'] = (df[longer] - df[shorter]) / df[signal]
    
    df = insert_fama_french_column(df)
    df["Log_Return_shift"] = df["Log_Return"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

def get_one_stock_features_df(stock_name,start_date,end_date,interval,trading_days_per_year, hours_per_day):
    """
    overall function to download data for specified stock and get features
    """
    stock_df = download_data_from_yf(stock_name,start_date,end_date,interval)
    stock_df = generate_features(stock_df, trading_days_per_year, hours_per_day)
    return stock_df

def get_all_stock_features_df(stocks_list,start_date,end_date,interval,trading_days_per_year, hours_per_day):
    """
    overall function to get full df of features df for a specified list of stocks
    """

    df = pd.DataFrame()
    
    for stock_name in stocks_list:
        stock_df = get_one_stock_features_df(stock_name,start_date,end_date,interval,trading_days_per_year, hours_per_day)
        df = pd.concat([df, stock_df])
    
    return df

    


# # can add code here to make it into a runnable python script
# if __name__=="__main__":

#     # Set up argument parser
#     parser = argparse.ArgumentParser(description="<add description later>")
#     parser.add_argument('--stock_name', type=str, required=True, help='Ticker symbol of the stock')
#     parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
#     parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
#     parser.add_argument('--interval', type=str, required=True, choices=['1d', '1wk', '1mo'], help='Data interval (1d, 1wk, 1mo)')


#     # download stock_data
#     stock_data = download_data_from_yf(stock_name,start_date,end_date,interval)

