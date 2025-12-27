# linear soft-margin svm (primal)
# from-scratch implementation verified on separable toy data

import numpy as np 

rng = np.random.default_rng(0)

n = 50
sigma = 0.3
C = 1.0
lr = 1e-3
n_iters=100

X_pos = rng.normal(loc=[1,1], scale=sigma, size=(n, 2))
X_neg = rng.normal(loc=[-1,-1], scale=sigma, size=(n,2))

X = np.vstack((X_pos, X_neg))
y = np.hstack((np.ones(n), -np.ones(n)))

def objective(w,b,C,X,y):
    """
    Compute the primal soft-margin SVM objective.

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    b : float
        Bias term.
    C : float
        Regularization parameter controlling the hinge loss penalty.
    X : ndarray of shape (n, d)
        Input data matrix.
    y : ndarray of shape (n,)
        Binary labels in {-1, +1}.

    Returns
    -------
    float
        Value of the objective function:
        0.5 * ||w||^2 + C * sum_i max(0, 1 - y_i (w^T x_i + b)).
    """
    tot = 0
    for i in range(len(X)):
        tot += max(0,1-y[i]*(np.dot(w,X[i])+b))
    return 0.5*np.dot(w,w) + C*tot

def grad(w,b,C,X,y):
    """
    Compute the subgradient of the primal soft-margin SVM objective.

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    b : float
        Bias term.
    C : float
        Regularization parameter.
    X : ndarray of shape (n, d)
        Input data matrix.
    y : ndarray of shape (n,)
        Binary labels in {-1, +1}.

    Returns
    -------
    grad_w : ndarray of shape (d,)
        Subgradient of the objective with respect to w.
    grad_b : float
        Subgradient of the objective with respect to b.
    """

    totw = np.zeros(X.shape[1])
    totb = 0
    for i in range(len(X)):
        if (y[i]*(np.dot(w,X[i])+b) < 1):
            totw += X[i] * y[i]
            totb += y[i]
    gradw = w-C*totw
    gradb = -C*totb
    return gradw, gradb

def train(X,y,C,lr,n_iters):
    """
    Train a linear soft-margin SVM using primal subgradient descent.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Input data matrix.
    y : ndarray of shape (n,)
        Binary labels in {-1, +1}.
    C : float
        Regularization parameter.
    lr : float
        Learning rate for subgradient descent.
    n_iters : int
        Number of optimization iterations.

    Returns
    -------
    w : ndarray of shape (d,)
        Learned weight vector.
    b : float
        Learned bias term.
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    for i in range(n_iters):
        gradw,gradb = grad(w,b,C,X,y)
        w -= lr*gradw
        b -= lr*gradb
    return w,b

if __name__ == "__main__":
    w, b = train(X, y, C, lr, n_iters)
    y_hat = np.sign(X @ w + b)
    acc = (y_hat == y).mean()
    print(f"training accuracy: {acc:.3f}")
