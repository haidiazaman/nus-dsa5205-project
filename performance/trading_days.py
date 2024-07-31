import pandas as pd
import pandas_market_calendars as mcal

def calculate_trading_days(start_date, end_date):
    nasdaq = mcal.get_calendar('NASDAQ')
    trading_days = nasdaq.valid_days(start_date=start_date, end_date=end_date)
    return len(trading_days)

# Example
start_date = '2023-01-01'
end_date = '2023-12-31'

trading_days = calculate_trading_days(start_date, end_date)
print(f"Number of trading days between {start_date} and {end_date}: {trading_days}")