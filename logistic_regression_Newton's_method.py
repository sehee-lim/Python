import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# logistic regression with Newton's method
def logistic_regression(x, y, num_epochs):
    N, D = x.shape
    w = np.zeros(D)   # initialize
    ll_history = []   # storing log-likelihood values

    for epoch in range(num_epochs):
        # log-likelihood value
        log_likelihood = np.sum(y * np.log(sigmoid(x @ w)) + (1 - y) * np.log(1 - sigmoid(x @ w)))
        ll_history.append(log_likelihood)
        
        hessian = x.T @ np.diag(sigmoid(x @ w) * (1 - sigmoid(x @ w))) @ x
        gradient = x.T @ (sigmoid(x @ w) - y)
        
        # update parameter
        w = w - np.linalg.inv(hessian) @ gradient

    return w, ll_history

# load data
x = np.load('data/x.npy')
y = np.load('data/y.npy')

# apply bias trick
x = np.concatenate((np.ones([x.shape[0], 1]), x), axis = 1)

w, ll_history = logistic_regression(x, y, num_epochs = 10)
print('Coefficient', w)
print('Log-likelihood', ll_history[-1])

# visualization
def visualization(ll_history):
    plt.plot(ll_history, c = 'blue', marker = 'o', label = 'Logistic regression')
    plt.xlabel('Epoch')
    plt.ylabel('Log-likelihood')
    plt.legend()
    plt.show()
    