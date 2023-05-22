import numpy as np
import matplotlib.pyplot as plt

# softmax regression with batch gradient descent
def softmax_regression(x, y, lr, num_epochs):
    N, D = x.shape
    K = len(np.unique(y))
    w = np.zeros([D, K])   # initialize
    ll_history = []   # storing historical log-likelihood values

    for epoch in range(num_epochs):
          # softmax probabilities
          p = np.exp(x @ w) / np.sum(np.exp(x @ w), axis = 1, keepdims = True)
          
          # update parameter
          gradient = x.T @ (np.eye(K)[y] - p)
          w = w + lr * gradient
          
          # log-likelihood value
          log_likelihood = np.sum(np.eye(K)[y] * np.log(p))
          ll_history.append(log_likelihood)
          
    return w, ll_history


# predict labels
def predict(w, x):
    y_pred = np.argmax(x @ w, axis = 1)   # predict with the highest score
    return y_pred

# compute accuracy of prediction
def evaluate(preds, labels):
    return (preds == labels).mean()

# load data
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

# apply bias trick
x_train = np.concatenate((np.ones([x_train.shape[0], 1]), x_train), axis = 1)
x_test = np.concatenate((np.ones([x_test.shape[0], 1]), x_test), axis = 1)
w, ll_history = softmax_regression(x_train, y_train, lr = 2e-2, num_epochs = 300)

# visualization
def visualization(ll_history):
    plt.plot(ll_history, c = 'blue', marker = 'o', label = 'Softmax regression')
    plt.xlabel('Epoch')
    plt.ylabel('Log-likelihood')
    plt.legend()
    plt.show()
    
preds_train = predict(w, x_train)
train_acc = evaluate(preds_train, y_train)
print('Train accuracy: {:5.2f}%'.format(100 * train_acc))

preds_test = predict(w, x_test)
test_acc = evaluate(preds_test, y_test)
print('Test  accuracy: {:5.2f}%'.format(100 * test_acc))