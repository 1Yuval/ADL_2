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
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)


def gradient_descent(A_train, b_train, A_test, b_test):
    n, m = A_train.shape
    epsilon = 0.01  # step size
    delta = 1e-5    # stop conditions
    #initialize the parameters
    xk = np.zeros(m).reshape((-1,1))
    #objective function
    err_fun = lambda x: np.linalg.norm(A_train.dot(x) - b_train)**2
    TRAIN_ERR = []
    MSE = []
    #gradient descent
    grad = 2*A_train.T.dot(A_train.dot(xk) - b_train)
    while np.linalg.norm(grad) > delta:
        err = err_fun(xk)
        TRAIN_ERR.append(err)
        xk = xk - epsilon * grad
        grad = 2*A_train.T.dot(A_train.dot(xk) - b_train)
        # Evaluate on the test set
        # Prediction
        b_pred = A_test.dot(xk)
        mse = np.mean((b_test - b_pred)**2)
        MSE.append(mse)
        
        
    #plot the results with title and bar title
    #the error plot of this procedure, namely a graph of the error l2 |A@xk − b|**2 where xk is the point at step k.
    # show a graph of the train error jointly with the graph of the test error
    plt.plot(MSE), plt.plot(TRAIN_ERR)
    ERR = np.stack(TRAIN_ERR)
    plt.legend(['Test Error', 'Train Error'])
    
    plt.title('Test Error vs Train Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error') 
    plt.show()

gradient_descent(A_train, b_train, A_test, b_test)


