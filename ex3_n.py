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



def gradient_descent(A_train, A_test, b_train, b_test,index, epsilon = 1e-3, delta = 5):
    
    n, m = A_train.shape

    #initialize the parameters
    xk = np.zeros(m).reshape((-1,1))
    #objective function
    err_train_fun = lambda x: np.linalg.norm(A_train.dot(x) - b_train)**2/len(b_train)
    err_test_fun = lambda x: np.linalg.norm(A_test.dot(x) - b_test)**2/len(b_test)
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
        test_err = err_test_fun(xk)
        TEST_ERR.append(test_err)
    #plot the results with title and bar title
    #the error plot of this procedure, namely a graph of the error l2 |A@xk − b|**2 where xk is the point at step k.
    # show a graph of the train error jointly with the graph of the test error
    plt.plot(TEST_ERR, label='Test error')
    plt.plot(TRAIN_ERR, label='Train error')
    TRAIN_ERR = np.stack(TRAIN_ERR)
    plt.legend(['Test Error', 'Train Error'])
    
    plt.title(f'Test Error vs Train Error over iterations For Time {index+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Error') 
    plt.show()
    return xk, TRAIN_ERR, TEST_ERR

def plot_min_max_avg(train_errors, test_errors, str):
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label=f'{str} Train Error')  
    plt.plot(test_errors, label=f'{str} Test Error')  
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(f'{str} Train and Test Errors over Iterations')
    plt.grid(True)
    plt.show()
def main():
    # Repeat the process with different splits and average results
    all_train_errors = []
    all_test_errors = []
    
    for i in range(10):
        # Split the data into a training set and a testing set
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2)
        xk, train_errors, test_errors = gradient_descent(A_train, A_test, b_train, b_test, i)
        all_train_errors.append(train_errors)
        all_test_errors.append(test_errors)
        print(f'xk of time {i+1}/10 is {xk}')
    

    # Use masked arrays to handle sequences of different lengths
    min_length = min(len(seq) for seq in all_train_errors)
    new_train_errors = np.array([seq[:min_length] for seq in all_train_errors])
    new_test_errors = np.array([seq[:min_length] for seq in all_test_errors])
    avg_train_errors = np.ma.mean(new_train_errors, axis=0)  
    avg_test_errors = np.ma.mean(new_test_errors, axis=0)  
    min_train_errors = np.ma.min(new_train_errors, axis=0)  
    min_test_errors = np.ma.min(new_test_errors, axis=0) 
    max_train_errors = np.ma.max(new_train_errors, axis=0)  
    max_test_errors = np.ma.max(new_test_errors, axis=0) 

    plot_min_max_avg(avg_train_errors, avg_test_errors, 'Average')
    plot_min_max_avg(min_train_errors, min_test_errors, 'Minimum')
    plot_min_max_avg(max_train_errors, max_test_errors, 'Maximum')


if __name__ == "__main__":
    main()