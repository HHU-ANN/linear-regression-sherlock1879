# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

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
    X, y = read_data()
    alpha = 0.001
    step = 0.001
    w,b=np.zeros((X.shape[1],6)),0
    m=X.shape[0]
    for _ in range(200):
        y_hat=np.dot(X,w)+b
        dw = np.dot(X, (y_hat - y)) / m + alpha * p(w)
        db = np.sum(y_hat - y) / m
        w-= step * dw
        b -= step * db
    return np.dot(X,w)+b

#l1正则化项求导
def p(w):
    pl1=[elem for elem in w]
    for i in range(len(pl1)):
        if pl1[i]>0:
            pl1[i]=1
        elif pl1[i]<0:
            pl1[i]=-1
        else:
            pl1[i]=0
    pl1=np.array(pl1)
    pl1=pl1.reshape((-1,))
    return pl1



def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y