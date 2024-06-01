import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#load the diabetes dataset
diabetes_sklearn = load_diabetes()
A = diabetes_sklearn.data
b = diabetes_sklearn.target.reshape((-1,1))
#initial point, step size ϵ, and the stopping condition δ
n, m = A.shape
epsilon = 0.01  # step size
delta = 1e-5    # stop conditions
#initialize the parameters
xk = np.zeros(m).reshape((-1,1))
#objective function
err_fun = lambda x: np.linalg.norm(A.dot(x) - b)**2
ERR = []
#gradient descent
grad = 2*A.T.dot(A.dot(xk) - b)
while np.linalg.norm(grad) > delta:
    err = err_fun(xk)
    ERR.append(err)
    xk = xk - epsilon * grad
    grad = 2*A.T.dot(A.dot(xk) - b)

print(xk)
    
plt.plot(ERR)
plt.legend(['Test Error', 'Train Error'])
plt.title('Test Error vs Train Error')
plt.xlabel('Iteration')
plt.ylabel('Error') 
plt.show()




