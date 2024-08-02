import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def visualize_and_cluster(correlation_matrix, top_30_stocks, num_clusters=10):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(correlation_matrix)

    df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    df['Stock'] = top_30_stocks

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['PC1', 'PC2']])

    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
              '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']

    fig = go.Figure()

    for i, color in enumerate(colors):
        cluster_data = df[df['Cluster'] == i]
        fig.add_trace(go.Scatter(
            x=cluster_data['PC1'],
            y=cluster_data['PC2'],
            mode='markers+text',
            marker=dict(color=color, size=10),
            text=cluster_data['Stock'],
            textposition="top center",
            name=f'Cluster {i}',
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}'
        ))

    fig.update_layout(
        title='Stock Clusters based on Correlation',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        legend_title='Clusters',
        font=dict(size=12),
        showlegend=True,
        width=1200,
        height=800,
        hovermode='closest'
    )

    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')

    pio.write_html(fig, file='stock_selection/stock_clusters.html', auto_open=True)
    pio.write_image(fig, file='stock_selection/stock_clusters.png')

    return df

def select_stocks_from_clusters(df, expected_returns, num_clusters=10):
    selected_stocks = []
    for i in range(num_clusters):
        cluster_stocks = df[df['Cluster'] == i]['Stock']
        cluster_returns = expected_returns[cluster_stocks]
        selected_stocks.append(cluster_returns.idxmax())
    return selected_stocks

def main():
    # Load data
    top_30_stocks = pd.read_csv('stock_selection/top_30_stocks.csv')['Ticker'][:29].tolist()
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    stock_data = get_stock_data(top_30_stocks, start_date, end_date)

    # Calculate returns & correlation
    returns = stock_data.pct_change().dropna()
    correlation_matrix = returns.corr()
    expected_returns = returns.mean() * 251  # Annualized returns

    # Perform clustering and visualization
    clustered_df = visualize_and_cluster(correlation_matrix, top_30_stocks)

    selected_stocks = select_stocks_from_clusters(clustered_df, expected_returns)
    print("Top 10 stocks selected:")
    for stock in selected_stocks:
        print(f"{stock}")

    results = pd.DataFrame({
        'Stock': selected_stocks
    })
    results.to_csv('stock_selection/chosen_10_stocks.csv', index=False)
    print("\nSelected stocks saved to 'chosen_10_stocks.csv'")
    print("Interactive cluster visualization saved to 'stock_clusters.html' and 'stock_clusters.png'")

if __name__ == "__main__":
    main()