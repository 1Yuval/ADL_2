import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load the diabetes dataset
diabetes_sklearn = load_diabetes()
A = diabetes_sklearn.data
b = diabetes_sklearn.target.reshape((-1,1))
# Split the data into a training set and a testing set


def gradient_descent():
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2)
    n, m = A_train.shape
    epsilon = 1e-3 # step size
    delta = 1e-2   # stop conditions
    #initialize the parameters
    x0 = np.zeros(m).reshape((-1,1))
    #objective function
    train_err_fun = lambda x: np.linalg.norm(A_train.dot(x) - b_train)**2/len(b_train)
    test_err_fun = lambda x: np.linalg.norm(A_test.dot(x) - b_test)**2/len(b_test)
    TRAIN_ERR = []
    TEST_ERR = []
    curr_train_err = train_err_fun(x0)
    curr_test_err = test_err_fun(x0)
    TRAIN_ERR.append(curr_train_err)
    TEST_ERR.append(curr_test_err)
    #gradient descent
    grad = 2*A_train.T.dot(A_train.dot(x0) - b_train)
    xk = x0
    while np.linalg.norm(grad) > delta:
        xk = xk - epsilon * grad
        grad = 2*A_train.T.dot(A_train.dot(xk) - b_train)
        # Evaluate on the test set
        # Prediction
        train_err = train_err_fun(xk)
        test_err = test_err_fun(xk)
        TRAIN_ERR.append(train_err)
        TEST_ERR.append(test_err)

    #plot the results with title and bar title
    #the error plot of this procedure, namely a graph of the error l2 |A@xk − b|**2 where xk is the point at step k.
    # show a graph of the train error jointly with the graph of the test error
    plt.figure(figsize=(10, 6))
    plt.plot(TEST_ERR, label= 'Test Error')
    plt.plot(TRAIN_ERR, label= 'Train Error')    
    plt.title('Test Error vs Train Error over Iterations for Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Error |Ax - b|^2 / len(b)') 
    plt.legend()
    plt.grid(True)
    plt.show()

gradient_descent()


