# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X, y = read_data()
    m=X.shape[0]
    n=X.shape[1]
    Lambda=500
    weight = np.matmul(np.linalg.inv(np.matmul(X.T, X)+Lambda*np.identity(n)), np.matmul(X.T, y))
    y_pred=weight @ data
    return weight @ data

def lasso(data):
    X, y = read_data()
    y_col=y.reshape(-1,1)
    #X(404,6)
    #y(404,)这是行向量！！！
    #theta(6,1)
    alpha = 300
    epochs = 25000
    learning_rate = 1e-9
    m,n=X.shape
    theta = np.zeros((n,1))
    for i in range(epochs):
        gradient = (1/m)*np.dot(X.T, np.dot(X, theta) - y_col) + alpha * np.sign(theta)
        theta = theta - learning_rate * gradient
    y_pred = data@theta
    return y_pred

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
