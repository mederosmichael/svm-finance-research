# svm-finance-research

## Research Question
Do canonical momentum and volatility features contain out-of-sample predictive information about next-period return direction in high-frequency financial time series?

## Null Hypothesis
H₀: A linear soft-margin SVM trained on canonical momentum and volatility indicators provides no statistically significant improvement over naïve baseline classifiers in predicting next-period return direction under strict temporal evaluation.

## Approach
This project implements a linear support vector machine from first principles and evaluates its performance on financial time series using a frozen dataset and leakage-safe temporal splits. Momentum and volatility features are constructed over rolling windows and used to predict the sign of the next-period return. Model performance is compared against trivial baselines, including constant prediction, random guessing, and last-return sign heuristics. Statistical significance is assessed using label permutation tests.

## Experiment 01: Linear SVM on 15-Minute BTC–USD
- **Asset:** BTC–USD  
- **Frequency:** 15-minute bars  
- **Observations:** 5,389  
- **Features:** Momentum, Volatility  
- **Label:** Sign of next-period return  

### Results
| Model            | Accuracy | Balanced Accuracy |
|------------------|----------|-------------------|
| Linear SVM       | 0.491    | 0.500             |
| Always +1        | 0.509    | 0.500             |
| Coin Flip        | 0.477    | 0.477             |
| Last Return Sign | 0.460    | 0.460             |

Permutation testing yields a null distribution centered at balanced accuracy 0.50 with a p-value of 1.0, indicating no evidence of linear separability in the chosen feature space.

### Conclusion
Under a frozen intraday BTC–USD dataset and strict temporal evaluation, canonical momentum and volatility features do not exhibit linear separability for next-period return prediction. Performance is indistinguishable from chance and trivial baselines, supporting the null hypothesis.

A detailed write-up of this experiment is available in `writeup/experiment_01.md`.
