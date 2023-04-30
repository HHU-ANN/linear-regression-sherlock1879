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
    y_pred=weight @ data
    return weight @ data
 #计算总数据量
def lasso(data):
    x,y=read_data()#x(404,6)y(404,)
    epochs=100000
    Lambda=0.1
    a=0.0000000000001#learning_rate
    m=x.shape[0]
    #给x添加偏置项
    X = np.concatenate((np.ones((m,1)),x),axis=1)#(m,n+1)
    #计算总特征数
    n = X.shape[1]
    #初始化W的值,要变成矩阵形式
    W=np.mat(np.ones((n,1)))
    #X转为矩阵形式
    xMat = np.mat(X)
    #y转为矩阵形式，这步非常重要,且要是m x 1的维度格式
    yMat =np.mat(y.reshape(-1,1))
    #循环epochs次
    for i in range(epochs):
        gradient = xMat.T*(xMat*W-yMat)/m + Lambda * np.sign(W)
        # print("xMat.T*(xMat*W-yMat)/m")
        # print(xMat.T*(xMat*W-yMat)/m)
        # print("Lambda * np.sign(W):")
        # print(Lambda * np.sign(W))
        # print("gradient:")
        # print(gradient)
        W=W-a * gradient
    Data = np.concatenate([data, [1]])
    return np.dot(Data,W)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
