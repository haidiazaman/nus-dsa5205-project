import requests
import pandas as pd
import plotly.graph_objs as go

API_KEY = ''                # Replace with your API key
TICKER = 'AAPL'
INTERVAL = '30'             # Minute interval for data
START_DATE = '2023-01-01'  
END_DATE = '2023-09-30'     # 90 days date range :(

url = f'https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/{INTERVAL}/minute/{START_DATE}/{END_DATE}'
params = {
    'adjusted': 'true',
    'sort': 'asc',
    'limit': 50000,
    'apiKey': API_KEY
}

response = requests.get(url, params=params)
data = response.json()

if response.status_code == 200 and 'results' in data:
    df = pd.DataFrame(data['results'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={'v': 'Volume', 'vw': 'VWAP', 'o': 'Open', 'c': 'Close', 'h': 'High', 'l': 'Low', 't': 'Timestamp', 'n': 'Transactions'}, inplace=True)
    print(df.head(1))
    
    fig = go.Figure(data=[
        go.Candlestick(x=df['Timestamp'],
                       open=df['Open'],
                       high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       name='Candlestick')
    ])

    fig.update_layout(
        title=f'AAPL Stock Data ({INTERVAL}-minute intervals)',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False
    )

    fig.show()
else:
    print("Error fetching data:", data)
