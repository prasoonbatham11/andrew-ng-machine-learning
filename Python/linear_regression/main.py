import numpy as np
import pandas as pd

def cost(X, y, theta):
    m,n = np.shape(X)
    H = np.dot(X, theta)
    # size of theta = n*1
    # size of X = m*n
    # size of H = m*1


    # size of y = m*1
    J = (1/(2*m))*np.sum((H-y)**2)
    return J

def gradient_descent(X, y, theta, alpha, iters):
    m,n = np.shape(X)

    for _ in range(0, iters):
        H = np.dot(X, theta)
        grad = (X*(H-y)).sum(axis=0).reshape(n,1)
        grad = grad*(alpha/m)
        theta = theta - grad
    return theta

def feature_normalize(X):
    # X = m*n
    # np.mean(X, axis = 0) = 1*n
    # np.std(X, axis = 0) = 1*n
    m,n = X.shape
    mu = np.mean(X, axis = 0).reshape(1, n)
    sigma = np.std(X, axis = 0).reshape(1, n)
    X = (X-mu)/sigma
    return (X, mu, sigma)


data_train = pd.read_csv('ex1data2.csv')

X = np.array(data_train[['Size','Bedrooms']]) # m*2
y = np.array(data_train['Price']) # m*1
#X = X.reshape(X.shape[0], 1)
y = y.reshape(y.shape[0], 1)
m,n = X.shape

#print(X[0:10,:])
#print(y[0:10,:])

X,mu,sigma = feature_normalize(X)
#print(X[0:10, :])

X = np.concatenate((np.ones((m,1)), X), axis = 1)

m,n = X.shape

theta = np.zeros((n,1))

print("Cost with theta = 0 ",cost(X,y,theta))

new_theta = gradient_descent(X, y, theta, 0.03, 400)
print("New Theta ",new_theta)

#print(np.dot([1,7], new_theta)*10000)
temp = [1,1650,3]
print("mu, sigma ", mu, sigma)
temp[1] = (temp[1]-mu[0][0])/sigma[0][0]
temp[2] = (temp[2] - mu[0][1])/sigma[0][1]

print(np.dot(temp, new_theta))

