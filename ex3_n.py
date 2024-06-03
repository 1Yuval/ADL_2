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



def gradient_descent_w_mse(A_train, A_test, b_train, b_test,index, epsilon = 0.01, delta = 1e-5):
    
    n, m = A_train.shape

    #initialize the parameters
    xk = np.zeros(m).reshape((-1,1))
    #objective function
    err_train_fun = lambda x: np.linalg.norm(A_train.dot(x) - b_train)**2
    TRAIN_ERR = []
    TEST_ERR = []
    #gradient descent
    grad = 2*A_train.T.dot(A_train.dot(xk) - b_train)
    while np.linalg.norm(grad) > delta:
        xk = xk - epsilon * grad
        grad = 2*A_train.T.dot(A_train.dot(xk) - b_train)
        train_err = err_train_fun(xk)
        TRAIN_ERR.append(train_err)
        # Evaluate on the test set
        # Prediction
        b_pred = A_test.dot(xk)
        mse = np.mean((b_test - b_pred)**2)
        TEST_ERR.append(mse)
    #plot the results with title and bar title
    #the error plot of this procedure, namely a graph of the error l2 |A@xk − b|**2 where xk is the point at step k.
    # show a graph of the train error jointly with the graph of the test error
    plt.plot(TEST_ERR), plt.plot(TRAIN_ERR)
    TRAIN_ERR = np.stack(TRAIN_ERR)
    plt.legend(['Test Error', 'Train Error'])
    
    plt.title(f'Test Error vs Train Error For iteration ${index}')
    plt.xlabel('Iteration')
    plt.ylabel('Error') 
    plt.show()
    return xk, TRAIN_ERR, TEST_ERR


def main():
    # Repeat the process with different splits and average results
    all_train_errors = []
    all_test_errors = []
    
    for i in range(10):
        # Split the data into a training set and a testing set
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)
        xk, train_errors, test_errors = gradient_descent_w_mse(A_train, A_test, b_train, b_test, i)
        all_train_errors.append(train_errors)
        all_test_errors.append(test_errors)
        print(f'xk of Iteration {i+1}/10 is {xk}')
    
    avg_train_errors = np.mean(all_train_errors, axis=0)
    avg_test_errors = np.mean(all_test_errors, axis=0)
    min_train_errors = np.min(all_train_errors, axis=0)
    min_test_errors = np.min(all_test_errors, axis=0)
    max_train_errors = np.max(all_train_errors, axis=0)
    max_test_errors = np.max(all_test_errors, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_errors, label='Average Train Error')
    plt.plot(avg_test_errors, label='Average Test Error')
    plt.plot(min_train_errors, label='Minimum Train Error')
    plt.plot(min_test_errors, label='Minimum Test Error')
    plt.plot(max_train_errors, label='Maximum Train Error')
    plt.plot(max_test_errors, label='Maximum Test Error')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Average Min and Max Train and Test Errors over Iterations')
    plt.show()

if __name__ == "__main__":
    main()