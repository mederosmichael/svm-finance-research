# experiment 01
# btc-usd, 15-minute bars
# linear svm, walk-forward evaluation

import yfinance as yf
import src.svm_primal as svm
import datetime as dt
import numpy as np

ASSET = 'BTC-USD'
FREQUENCY = '15m'
L = 96

df = yf.download(
    tickers=ASSET, 
    start=dt.datetime.now() - dt.timedelta(days=59), 
    end=dt.datetime.now(), 
    interval=FREQUENCY
)

df['log_prices'] = np.log(df['Close'])
df['r'] = df['log_prices'].diff()
df['y'] = np.sign(df['r'].shift(-1))
df.loc[df['y']==0,'y'] = 1
df['Momentum'] = df['r'].rolling(L).sum()
df['rMean'] = df['r'].rolling(L).mean()
df['d'] = df['r'] - df['rMean']
df["d2"] = df["d"] ** 2
df["Volatility"] = np.sqrt(df["d2"].rolling(L).mean())
df = df.dropna()

X = df[["Momentum", "Volatility"]].values
y = df['y'].values
C = 1.0
lr = 1e-3
T = 1000

w,b = svm.train(X,y,C,lr,T)