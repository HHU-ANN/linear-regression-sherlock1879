# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
import copy

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X, y = read_data()
    weight = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
    return weight @ data

def lasso(data):
    x, y = read_data()
    m = x.shape[0]
    epochs=100

    X = np.concatenate((np.ones((m, 1)), x), axis=1)
    xMat = np.mat(X)
    yMat = np.mat(y.reshape(-1, 1))

    w = np.ones(X.shape[1]).reshape(-1, 1)

    for n in range(epochs):

        out_w = copy.copy(w)
        for i, item in enumerate(w):
            # 在每一个W值上找到使损失函数收敛的点
            for j in range(epochs):
                h = xMat * w
                gradient = xMat[:, i].T * (h - yMat) / m + Lambda * np.sign(w[i])
                w[i] = w[i] - gradient * learning_rate
                if abs(gradient) < 1e-3:
                    break
        out_w = np.array(list(map(lambda x: abs(x) < 1e-3, out_w - w)))
        if out_w.all():
            break
    return np.dot(X,w)



def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y