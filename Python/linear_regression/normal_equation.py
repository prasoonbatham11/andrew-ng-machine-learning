import pandas as pd
import numpy as np

data_train = pd.read_csv('ex1data2.csv')

X = np.array(data_train[['Size','Bedrooms']]) # m*2
y = np.array(data_train['Price']) # m*1
m,n = X.shape
y = y.reshape(y.shape[0], 1)

X = np.concatenate((np.ones((m,1)), X), axis = 1)

h = np.dot(X.T,X)
h = np.linalg.inv(h)
h = np.dot(h, X.T)
h = np.dot(h, y)
print(h)

print(np.dot([1,1650,3], h))