import pandas as pd
import numpy as np

data_train = pd.read_csv('ex2data1.csv')

X = np.array(data_train[['Maths','Physics']]) # m*2
y = np.array(data_train['Admitted']) # m*1
y = y.reshape(y.shape[0], 1)
m,n = X.shape
X = np.concatenate(( np.ones((m,1)), X ), axis=1)
n+=1
theta = np.zeros((n,1))

def sigmoid(t):
    g = 1/(1+np.exp(-t))
    return g

def cost(X, y, theta):
    m,n = X.shape
    H = sigmoid(np.dot(X, theta))
    J = -(1/m)*(np.sum(y*np.log(H) + (1-y)*np.log(1-H)))
    return J

def gradient_descent(X, y, theta, alpha, iters):
    m,n = np.shape(X)

    for _ in range(0, iters):
        H = sigmoid(np.dot(X, theta))
        grad = (X*(H-y)).sum(axis=0).reshape(n,1)
        grad = grad*alpha
        theta = theta - grad
    return theta


J = cost(X,y,theta)

newtheta = gradient_descent(X, y, theta, 0.00005, 400)
print(newtheta)

pre = sigmoid( np.dot([1,45,85] , newtheta) )
print(pre)

