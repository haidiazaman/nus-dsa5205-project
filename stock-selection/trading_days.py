import pandas_market_calendars as mcal

def calculate_trading_days(start_date, end_date, market):
    if market == 'NASDAQ':
        corrected_market = 'NASDAQ'
    elif market == 'FTSE': 
        corrected_market = 'LSE'
    elif market == 'SGX':
        corrected_market = 'XSES'
    else: 
        return 251

    market = mcal.get_calendar(corrected_market)
    trading_days = market.valid_days(start_date=start_date, end_date=end_date)
    return len(trading_days)

# Example
start_date = '2023-01-01'
end_date = '2023-12-31'

trading_days = calculate_trading_days(start_date, end_date, 'SGX')
print(f"Number of trading days between {start_date} and {end_date}: {trading_days}")