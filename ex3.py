import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
diabetes_sklearn = load_diabetes()
A = diabetes_sklearn.data
b = diabetes_sklearn.target.reshape((-1, 1))

def gradient_descent(A_train, b_train, epsilon=0.01, delta=1e-5, max_iter=10000):
    n, m = A_train.shape
    xk = np.zeros((m, 1))
    err_fun = lambda x: np.linalg.norm(A_train.dot(x) - b_train)**2
    grad = 2 * A_train.T.dot(A_train.dot(xk) - b_train)
    
    errors = []
    while np.linalg.norm(grad) > delta and len(errors) < max_iter:
        xk = xk - epsilon * grad
        grad = 2 * A_train.T.dot(A_train.dot(xk) - b_train)
        error = err_fun(xk)
        errors.append(error)
    return xk, errors

def evaluate_model(A_train, b_train, A_test, b_test, epsilon=0.01, delta=1e-5):
    xk, train_errors = gradient_descent(A_train, b_train, epsilon, delta)
    train_errors = np.array(train_errors)
    test_errors = []
    for i in range(len(train_errors)):
        b_pred = A_test.dot(xk)
        test_error = np.mean((b_test - b_pred)**2)
        test_errors.append(test_error)
    
    return train_errors, test_errors, xk

def main():
    # Split the data into a training set and a testing set
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)

    # Evaluate the model and plot errors
    train_errors, test_errors, xk = evaluate_model(A_train, b_train, A_test, b_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label='Train Error')
    plt.plot(test_errors, label='Test Error')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Train and Test Error over Iterations')
    plt.show()

    # Repeat the process with different splits and average results
    all_train_errors = []
    all_test_errors = []

    for i in range(10):
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2)
        train_errors, test_errors, xk = evaluate_model(A_train, b_train, A_test, b_test)
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
