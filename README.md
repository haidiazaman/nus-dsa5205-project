# nus-dsa5205-project
this repo consolidates the codes and development work done for the NUS DSA5205, Data Science in Quantitative Finance module, completed in AY23/24 STII.

* Research Objectives: Design an automated trading algorithm. It consists of stock selection, ML-based signal-generation and trade execution based on the signals. The final result can be used by retail traders who like to invest in the stock market but do not
have time to follow the market closely. 

* Approach Chosen: Choose the 10 stocks from the investment universe using random forest classifier; basic time series analysis on data; feature engineering based on on a grid of 72 features (derived from technical indicators such as VWAP and EMAs); train a fully connected neural network (FCN) to generate trading signals (long, neutral and short signals); rebalance the portfolio with equal-weight.

* Stock Selection: Our stock universe comprised NASDAQ 100, FTSE 100, and 20 SGX stocks, balancing liquidity, diversification and returns.
    * Applied PCA to the correlation matrix of daily returns (2022 - 2023), reducing dimensionality to two components explaining 72.94% of total variance. ![alt text](https://github.com/haidiazaman/nus-dsa5205-project/blob/main/imgs/pca.png)
    *  Used K-means clustering to group stocks into 10 clusters based on PCA coordinates.
    *  Selected the stock with the highest expected annualised return from each cluster. Our final 10 stocks are: DDOG, SNPS, BKNG, SMCI, MDB, NVDA, MELI, WDAY, MU, PDD, with MU replacing RR.L to avoid time zone complications. ![alt text](https://github.com/haidiazaman/nus-dsa5205-project/blob/main/imgs/stock_return.png)

* Dataset Time Series Analysis
    * The returns reveal stationarity (confirmed by ADF test), but ACF and PACF plots indicate no significant autocorrelation. This suggests that linear time series models like ARMA or ARCH are not suitable for modelling these returns or their volatility; advanced approaches are needed to model the potential non-linearity.![alt text](https://github.com/haidiazaman/nus-dsa5205-project/blob/main/imgs/ts_analysis.png)

* Methods
    * Markowitz Portfolio Optimisation ![alt text](https://github.com/haidiazaman/nus-dsa5205-project/blob/main/imgs/markowitz.png)
    * Fully-connected Neural Networks (FCN) ![alt text](https://github.com/haidiazaman/nus-dsa5205-project/blob/main/imgs/fcn.png)

* Comparison of results
  
![alt text](https://github.com/haidiazaman/nus-dsa5205-project/blob/main/imgs/comparison.png) 
