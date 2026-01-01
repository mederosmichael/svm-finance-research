# Experiment 01: Linear SVM on Short-Horizon BTC Returns

## Abstract
This experiment tests whether short-horizon Bitcoin returns exhibit a linearly separable signal under canonical momentum and volatility features. Using a frozen 15-minute BTC–USD dataset comprising 5,389 observations, we train a linear support vector machine to predict the sign of the next-period return. Evaluation follows a strict temporal split with no shuffling or tuning on test data. Model performance matches trivial baselines, achieving balanced accuracy equal to chance. A permutation test confirms that the observed result is statistically indistinguishable from randomness.

## 1. Research Question
Do momentum and volatility features admit a linearly separable signal for predicting the direction of the next 15-minute BTC–USD return?

The null hypothesis states that no such linear separation exists at this horizon and feature resolution.

## 2. Data
- **Asset:** BTC–USD  
- **Frequency:** 15-minute bars  
- **Dataset type:** Frozen snapshot (no re-downloading during experiments)  
- **Start:** 2025-11-05 20:30 UTC  
- **End:** 2026-01-01 20:30 UTC  
- **Observations:** 5,389  
- **Columns:** Open, High, Low, Close, Adj Close, Volume  

All experiments load exclusively from this fixed dataset to ensure full reproducibility.

## 3. Method

### 3.1 Label
The target label is the sign of the next-period log return:
- Log prices computed from the close.
- Returns defined as first differences of log prices.
- Label defined as `sign(r_{t+1})`, with zeros mapped to +1.

### 3.2 Features
Two canonical features are constructed using a rolling window of length `L = 96`:
- **Momentum:** rolling sum of returns.
- **Volatility:** square root of the rolling mean of squared demeaned returns.

### 3.3 Model
A linear support vector machine is trained via a primal formulation with:
- Regularization parameter `C = 1.0`
- Learning rate `lr = 1e-3`
- Training iterations `T = 1000`

No hyperparameter tuning is performed.

### 3.4 Evaluation
The dataset is split temporally:
- First 80% used for training
- Final 20% used for testing

Metrics reported:
- Accuracy
- Balanced accuracy

### 3.5 Baselines
Three baselines provide context:
1. Always predict +1  
2. Random coin flip matched to training class balance  
3. Sign of the previous return  

### 3.6 Permutation Test
Labels in the training set are permuted 200 times. The model is retrained on each permutation to form a null distribution of balanced accuracy. The p-value is computed as the fraction of null scores exceeding the observed score.

## 4. Results

| Model            | Accuracy | Balanced Accuracy |
|------------------|----------|-------------------|
| Linear SVM       | 0.491    | 0.500             |
| Always +1        | 0.509    | 0.500             |
| Coin Flip        | 0.477    | 0.477             |
| Last Return Sign | 0.460    | 0.460             |

Permutation test results:
- **Mean null balanced accuracy:** 0.500  
- **p-value:** 1.0  

## 5. Interpretation
The linear SVM achieves balanced accuracy equal to chance and does not outperform trivial baselines. The permutation test shows that the observed performance is entirely typical under random labeling. These results indicate that, at a 15-minute horizon, momentum and volatility features do not yield a linearly separable signal for BTC return direction.

The most plausible explanations are high microstructure noise at short horizons and the absence of linear structure in the chosen feature space. Any exploitable signal at this resolution likely requires nonlinear interactions, regime conditioning, or alternative label definitions.

## 6. Limitations and Next Steps
This experiment tests only linear separability under a narrow feature set and a single prediction horizon. Future work may explore longer horizons, no-trade thresholds, or nonlinear models. Such extensions should be treated as separate experiments and evaluated under the same frozen-data discipline.

## Conclusion
Under strict temporal evaluation on a frozen intraday BTC–USD dataset, momentum and volatility features do not exhibit linear separability for next-period return prediction. Performance matches chance and trivial baselines, with strong statistical confirmation of the null hypothesis.
