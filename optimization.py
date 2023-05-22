import numpy as np
import math
import matplotlib.pyplot as plt

def rms_error(y_true, y_pred = None):
    # compute the root mean square error
    if y_pred is None:
        return np.sqrt(np.mean(y_true ** 2))
    else:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


def polynomial_features(x, M):
    N = x.shape[0]
    phi = np.zeros((N, M + 1))
    phi[:, 0] = np.ones(N)
    
    for i in range(1, M+1):
        phi[:, i] = x ** i

    return phi

# batch gradient descent
def bgd(x_train, y_train, x_test, y_test, lr, num_epochs, M = 1, lam = 0):
    w = np.zeros(M + 1)   # initialize
    train_error = test_error = 0
    N = x_train.shape[0]
    rmse_list = []

    if M == 1:
        # add intercept term
        with_intercept = np.vstack((np.ones(N), x_train)).T
        with_intercept_test = np.vstack((np.ones(x_test.shape[0]), x_test)).T

        epoch = 0
        while epoch < num_epochs:
            pred = with_intercept @ w.T
            gradient = with_intercept.T @ (pred - y_train) + lam * w
            w = w - lr * gradient / N   # update w

            # compute rmse
            rmse = rms_error(y_train, pred)
            rmse_list.append(rmse)   # add to the rmse_list

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]   # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test @ w)


    elif M > 1:
        with_intercept = polynomial_features(x_train, M)
        with_intercept_test = polynomial_features(x_test, M)

        epoch = 0
        while epoch < num_epochs:
            pred = with_intercept @ w.T
            gradient = with_intercept.T @ (pred - y_train) + lam * w
            w = w - lr * gradient / N  # update w

            # compute rmse
            rmse = rms_error(y_train, pred)
            rmse_list.append(rmse)  # add to the rmse_list

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]  # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test @ w)

    # M == 0
    else:
        with_intercept = np.ones(N).T
        with_intercept_test = np.ones(N).T

        epoch = 0
        while epoch < num_epochs:
            pred = with_intercept.reshape((-1, 1)) @ w.T
            gradient = with_intercept.T @ (pred - y_train)
            w = w - lr * gradient / N  # update w

            # compute rmse
            rmse = rms_error(y_train, pred)
            rmse_list.append(rmse)  # add to the rmse_list

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]  # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test.reshape((-1, 1)) @ w)

    return w, train_error, test_error, rmse_list

# stochastic gradient descent
def sgd(x_train, y_train, x_test, y_test, lr, num_epochs, M = 1, lam = 0):
    w = np.zeros(M + 1)   # initialize
    train_error = test_error = 0
    N = x_train.shape[0]
    rmse_list = []

    if M == 1:
        # add intercept term
        with_intercept = np.vstack((np.ones(N), x_train)).T
        with_intercept_test = np.vstack((np.ones(x_test.shape[0]), x_test)).T

        epoch = 0
        while epoch < num_epochs:

            # to choose single training data
            index = np.random.randint(N, size = 1)
            pred = with_intercept[index] @ w
            
            # use only one training data when computing gradient
            gradient = (y_train[index] - pred) * with_intercept[index] + lam * w
            w = w + lr * gradient/N   # update
            w = w.reshape(-1)   # change dimension

            # compute training rmse
            rmse = rms_error(y_train, with_intercept @ w)
            rmse_list.append(rmse)

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]   # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test @ w.T)


    elif M > 1:
        with_intercept = polynomial_features(x_train, M)
        with_intercept_test = polynomial_features(x_test, M)

        epoch = 0
        while epoch < num_epochs:
            # to choose single training data
            index = np.random.randint(N, size = 1)
            pred = with_intercept[index] @ w
            
            # use only one training data when computing gradient
            gradient = (y_train[index] - pred) * with_intercept[index] + lam * w
            w = w + lr * gradient / N  # update
            w = w.reshape(-1)  # change dimension

            # compute training rmse
            rmse = rms_error(y_train, with_intercept @ w)
            rmse_list.append(rmse)

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]  # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test @ w.T)

    # M == 0
    else:
        with_intercept = np.ones(N).T
        with_intercept_test = np.ones(N).T

        epoch = 0
        while epoch < num_epochs:

            # to choose single training data
            index = np.random.randint(N, size = 1)
            pred = with_intercept[index] @ w
            
            # use only one training data when computing gradient
            gradient = (y_train[index] - pred) * with_intercept[index]
            w = w + lr * gradient / N  # update
            w = w.reshape(-1)  # change dimension

            # compute training rmse
            rmse = rms_error(y_train, with_intercept.reshape((-1, 1)) @ w)
            rmse_list.append(rmse)

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]  # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test.reshape((-1, 1)) @ w.T)
        
    return w, train_error, test_error, rmse_list

# Newton's method
def newton(x_train, y_train, x_test, y_test, lr, num_epochs, M=1, lam=0.):
    w = np.zeros(M + 1)   # initialize
    train_error = test_error = 0
    N = x_train.shape[0]
    rmse_list = []

    if M == 1:
        # add intercept term
        with_intercept = np.vstack((np.ones(N), x_train)).T
        with_intercept_test = np.vstack((np.ones(x_test.shape[0]), x_test)).T

        epoch = 0
        while epoch < num_epochs:
            pred = with_intercept @ w.T
            gradient = with_intercept.T @ (pred - y_train) + lam * w
            hessian = np.dot(with_intercept.T, with_intercept) + np.eye(M + 1) * lam

            w = w - np.linalg.inv(hessian) / N @ gradient / N  # update w

            # compute rmse
            rmse = rms_error(y_train, pred)
            rmse_list.append(rmse)  # add to the rmse_list

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]  # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test @ w)

    elif M > 1:
        with_intercept = polynomial_features(x_train, M)
        with_intercept_test = polynomial_features(x_test, M)

        epoch = 0
        while epoch < num_epochs:
            pred = with_intercept @ w.T
            gradient = with_intercept.T @ (pred - y_train) + lam * w
            hessian = np.dot(with_intercept.T, with_intercept) + np.eye(M + 1) * lam

            w = w - np.linalg.inv(hessian)/N @ gradient/N  # update w

            # compute rmse
            rmse = rms_error(y_train, pred)
            rmse_list.append(rmse)  # add to the rmse_list

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]  # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test @ w)

    else:
        with_intercept = np.ones(N).T
        with_intercept_test = np.ones(N).T

        epoch = 0
        while epoch < num_epochs:

            pred = with_intercept.reshape((-1, 1)) @ w.T
            gradient = with_intercept.T @ (pred - y_train)
            hessian = np.dot(with_intercept.T, with_intercept)

            w = w - gradient / hessian  # update w

            # compute rmse
            rmse = rms_error(y_train, pred)
            rmse_list.append(rmse)  # add to the rmse_list

            # if rmse is low enough, terminate while loop
            if rmse <= 0.2:
                break

            epoch = epoch + 1

        train_error = rmse_list[-1]  # the last element is the training error

        # compute test rmse
        test_error = rms_error(y_test, with_intercept_test.reshape((-1, 1)) @ w)

    return w, train_error, test_error, rmse_list


x_train = np.load('data/xTrain.npy')
y_train = np.load('data/yTrain.npy')
x_test = np.load('data/xTest.npy')
y_test = np.load('data/yTest.npy')
num_epochs = 1000

# batch gradient descent
w, train_error, test_error, rmse_list_bgd = bgd(x_train, y_train, x_test, y_test, lr = 0.04, num_epochs = num_epochs)
print('Coefficient generated by BGD', w)
print('Train RMS error: ', train_error)
print('Test  RMS error: ', test_error)

# stochastic gradient descent
w, train_error, test_error, rmse_list_sgd = sgd(x_train, y_train, x_test, y_test, lr = 0.8, num_epochs = num_epochs)
print('Coefficient generated by SGD', w)    
print('Train RMS error: ', train_error)
print('Test  RMS error: ', test_error)

# Newton's method
w, train_error, test_error, rmse_list_newton = newton(x_train, y_train, x_test, y_test, lr = 0.03, num_epochs = num_epochs)
print('Coefficient generated by Newton\'s method', w)
print('Train RMS error: ', train_error)
print('Test  RMS error: ', test_error)


# visualization
def visualization(rmse_list_bgd, rmse_list_sgd, rmse_list_newton):
    plt.plot(rmse_list_bgd, c = 'blue', marker = 'o', label = 'BGD')
    plt.plot(rmse_list_sgd, c = 'red', marker = 'o', label = 'SGD')
    plt.plot(rmse_list_newton, c = 'black', marker = 'o', label = 'Newton')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
