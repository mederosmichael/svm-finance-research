# experiment 01
# btc-usd, 15-minute bars
# linear svm, walk-forward evaluation

import pandas as pd
import src.svm_primal as svm
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

print("rows:", len(df))
print("start:", df.index.min())
print("end:", df.index.max())
print("features: Momentum, Volatility")
print("label: sign(next return)")

print("test balanced accuracy:", bacc)
print("test class balance (+1):", np.mean(pos))
# =====================
# baselines
# =====================

y_hat_allpos = np.ones_like(y_test)
acc_allpos = np.mean(y_hat_allpos == y_test)

tpr_allpos = np.mean(y_hat_allpos[pos] == 1) if np.any(pos) else np.nan
tnr_allpos = np.mean(y_hat_allpos[neg] == -1) if np.any(neg) else np.nan
bacc_allpos = 0.5 * (tpr_allpos + tnr_allpos)

print("baseline all +1 accuracy:", acc_allpos)
print("baseline all +1 balanced accuracy:", bacc_allpos)


p = np.mean(y_train == 1)
rng = np.random.default_rng(0)
y_hat_coin = np.where(rng.random(len(y_test)) < p, 1, -1)

acc_coin = np.mean(y_hat_coin == y_test)
tpr_coin = np.mean(y_hat_coin[pos] == 1) if np.any(pos) else np.nan
tnr_coin = np.mean(y_hat_coin[neg] == -1) if np.any(neg) else np.nan
bacc_coin = 0.5 * (tpr_coin + tnr_coin)

print("baseline coin accuracy:", acc_coin)
print("baseline coin balanced accuracy:", bacc_coin)


y_hat_lastret = np.sign(df_test["r"].values)
y_hat_lastret[y_hat_lastret == 0] = 1

acc_lastret = np.mean(y_hat_lastret == y_test)
tpr_lastret = np.mean(y_hat_lastret[pos] == 1) if np.any(pos) else np.nan
tnr_lastret = np.mean(y_hat_lastret[neg] == -1) if np.any(neg) else np.nan
bacc_lastret = 0.5 * (tpr_lastret + tnr_lastret)

print("baseline last-return accuracy:", acc_lastret)
print("baseline last-return balanced accuracy:", bacc_lastret)
# =====================
# permutation test
# =====================

rng = np.random.default_rng(0)
B = 50
null_bacc = []

for _ in range(B):
    y_perm = rng.permutation(y_train)
    w_p, b_p = svm.train(X_train, y_perm, C, lr, T)

    scores_p = X_test @ w_p + b_p
    y_hat_p = np.sign(scores_p)
    y_hat_p[y_hat_p == 0] = 1

    tpr_p = np.mean(y_hat_p[pos] == 1) if np.any(pos) else np.nan
    tnr_p = np.mean(y_hat_p[neg] == -1) if np.any(neg) else np.nan
    null_bacc.append(0.5 * (tpr_p + tnr_p))

null_bacc = np.array(null_bacc)
p_value = (np.sum(null_bacc >= bacc) + 1) / (B + 1)

print("perm test mean null bacc:", float(null_bacc.mean()))
print("perm test p-value (>= observed):", float(p_value))
