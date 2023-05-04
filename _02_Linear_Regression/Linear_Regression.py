# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x, y = read_data()
    degree=2
    alpha=0.1
    # 增加多项式特征
    x_poly = np.ones_like(x)
    for d in range(1, degree + 1):
        x_poly = np.hstack((x_poly, np.power(x, d)))

    # 添加正则化项
    xtx = np.dot(x_poly.T, x_poly)
    reg_term = alpha * np.eye(xtx.shape[0])
    xtx_reg = xtx + reg_term

    # 求解系数
    xty = np.dot(x_poly.T, y)
    w = np.linalg.solve(xtx_reg, xty)

    # 预测结果
    x_new_poly = np.ones((1, x.shape[1]))
    for d in range(1, degree + 1):
        x_new_poly = np.hstack((x_new_poly, np.power(x, d)))
    y_pred = np.dot(x_new_poly, w)
    return y_pred

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
