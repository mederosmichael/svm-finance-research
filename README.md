# svm-finance-research
**Research Question
**
Do canonical momentum and volatility features contain out-of-sample predictive information about next-period return direction in financial time series?

**Null Hypothesis**

H₀: A linear soft-margin SVM trained on canonical momentum and volatility indicators provides no statistically significant improvement over naïve baseline classifiers in predicting next-period return direction under a walk-forward time-series evaluation.

**Approach**

This project implements a linear SVM from first principles and evaluates its out-of-sample performance on financial time series using strict walk-forward splits. Canonical momentum and volatility features are used, and results are validated against baseline classifiers and label-permutation tests to detect spurious structure and leakage.
