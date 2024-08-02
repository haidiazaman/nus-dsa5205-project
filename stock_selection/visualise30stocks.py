import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import warnings
import logging
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_tickers(file_path):
    return pd.read_csv(file_path)['Ticker'].tolist()

def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns_and_risk(prices):
    results = []
    
    for ticker in prices.columns:
        try:
            price_series = prices[ticker].dropna()
            
            if len(price_series) < 2:
                logging.warning(f"Insufficient data for {ticker}. Skipping.")
                continue
            
            start_price = price_series.iloc[0]
            end_price = price_series.iloc[-1]
            annual_return = (end_price / start_price) - 1
            daily_returns = price_series.pct_change().dropna()
            annualized_volatility = daily_returns.std() * np.sqrt(251)  # Assuming 251 trading days in a year
            
            results.append({
                'Ticker': ticker,
                'Annual Return': annual_return,
                'Annualized Volatility': annualized_volatility
            })
            
            logging.info(f"{ticker}: Annual Return = {annual_return:.2%}, Annualized Volatility = {annualized_volatility:.2%}")
        
        except Exception as e:
            logging.error(f"Error calculating returns and risk for {ticker}: {e}")
    
    returns_risk_df = pd.DataFrame(results).set_index('Ticker')
    
    return returns_risk_df

def plot_risk_return_scatter(returns):
    fig = px.scatter(returns, x='Annualized Volatility', y='Annual Return', text=returns.index,
                     labels={'Annualized Volatility': 'Annualized Volatility', 'Annual Return': 'Annual Return (2023)'},
                     title='Risk-Return Profile of Selected Stocks (2023)')
    
    fig.update_traces(textposition='top center')
    fig.update_layout(width=800, height=600)
    return fig

def plot_stock_correlation(returns):
    corr_matrix = returns.corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.index,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    
    fig.update_layout(
        title='Stock Return Correlation Heatmap',
        title_font_size=20,
        width=800,
        height=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title=None,
        yaxis_title=None
    )
    
    fig.update_xaxes(tickangle=45)
    return fig

def get_exchange_rate(from_currency, to_currency):
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    response = requests.get(url)
    data = response.json()
    return data['rates'][to_currency]

def get_market_cap(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if not hist.empty:
            last_close = hist['Close'].iloc[-1]
            shares_outstanding = stock.info.get('sharesOutstanding')
            
            if shares_outstanding is None:
                logging.warning(f"Shares outstanding not available for {ticker}")
                return np.nan
            
            market_cap = last_close * shares_outstanding
        
            if ticker.endswith('.L'):
                gbp_usd_rate = get_exchange_rate('GBP', 'USD')
                market_cap *= gbp_usd_rate
                market_cap /= 100  
                
            return market_cap
        else:
            logging.warning(f"No historical data available for {ticker}")
            return np.nan
    except Exception as e:
        logging.error(f"Error fetching market cap for {ticker}: {e}")
        return np.nan

def plot_market_cap_treemap(tickers, start_date, end_date):
    market_caps = {ticker: get_market_cap(ticker, start_date, end_date) for ticker in tickers}
    df = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'Market Cap'])
    df = df.dropna().sort_values('Market Cap', ascending=False)
    
    fig = px.treemap(df, path=['Ticker'], values='Market Cap',
                     title=f'Market Capitalization of Selected Stocks (as of last available date in {end_date})')
    fig.update_layout(width=800, height=600)
    return fig

def plot_feature_importance(feature_importance):
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                 title='Feature Importance in Stock Selection Model')
    
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        width=800,
        height=600,
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

tickers = read_tickers('stock_selection/top_30_stocks.csv')
start_date = '2023-01-01'
end_date = '2023-12-29'
market_cap_start_date = '2023-12-22' 
market_cap_end_date = '2023-12-29'

logging.info(f"Fetching data for date range: {start_date} to {end_date}")
logging.info(f"Using market cap date range: {market_cap_start_date} to {market_cap_end_date}")

prices = fetch_stock_data(tickers, start_date, end_date)
returns_risk = calculate_returns_and_risk(prices)

for ticker in returns_risk.index:
    logging.info(f"{ticker}: Annual Return = {returns_risk.loc[ticker, 'Annual Return']:.2%}, Annualized Volatility = {returns_risk.loc[ticker, 'Annualized Volatility']:.2%}")

fig1 = plot_stock_correlation(prices.pct_change().dropna())
fig2 = plot_market_cap_treemap(tickers, market_cap_start_date, market_cap_end_date)
fig3 = plot_risk_return_scatter(returns_risk)

feature_importance = pd.read_csv('feature_importance.csv')
fig4 = plot_feature_importance(feature_importance)

fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'xy'}, {'type': 'xy'}],
                                           [{'type': 'domain'}, {'type': 'xy'}]],
                    subplot_titles=('Stock Return Correlation', 'Risk-Return Profile',
                                    'Market Capitalization', 'Feature Importance'))

for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)
for trace in fig3.data:
    fig.add_trace(trace, row=1, col=2)
    fig.add_trace(fig2.data[0], row=2, col=1)
for trace in fig4.data:
    fig.add_trace(trace, row=2, col=2)

fig.update_layout(height=1600, width=1600, title_text="Visualise 30 Stocks")

fig.write_html("stock_selection/visualise_30_stocks.html")
print("Dashboard saved as 'visualise_30_stocks.html'")