# experiment 01
# btc-usd, 15-minute bars
# linear svm, walk-forward evaluation

import pandas as pd
import src.svm_primal as svm
import datetime as dt
import numpy as np
from pathlib import Path

#ASSET = 'BTC-USD'
#FREQUENCY = '15m'
L = 96
C = 1.0
lr = 1e-3
T = 1000
DATA_PATH = Path("data/BTC-USD_15m_20251103_20260101.csv")

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

if df.empty:
    raise RuntimeError("loaded empty dataframe")

for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Close"])



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

N = len(df)
split = int(0.8 * N)

df_train = df.iloc[:split]
df_test = df.iloc[split:]

X_train = df_train[["Momentum", "Volatility"]].values
y_train = df_train['y'].values

X_test = df_test[["Momentum", "Volatility"]].values
y_test = df_test['y'].values

w,b = svm.train(X_train,y_train,C,lr,T)

scores = X_test @ w + b
y_hat = np.sign(scores)
y_hat[y_hat == 0] = 1

acc = np.mean(y_hat == y_test)
print("test accuracy:", acc)

pos = (y_test == 1)
neg = (y_test == -1)

tpr = np.mean(y_hat[pos] == 1) if np.any(pos) else np.nan
tnr = np.mean(y_hat[neg] == -1) if np.any(neg) else np.nan
bacc = 0.5 * (tpr + tnr)

print("test balanced accuracy:", bacc)
print("test class balance (+1):", np.mean(pos))
