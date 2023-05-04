# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

try:
    import sklearn
except ImportError as e:
    os.system("sudo pip3 install scikit-learn")
    import sklearn

def ridge(data):
    x, y = read_data()
    # 将x转换为n列特征的矩阵，其中n为多项式的阶数
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(x)

    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # 创建一个线性回归模型
    model = LinearRegression()

    # 拟合训练数据
    model.fit(X_train, y_train)

    y_pred=model.predict(data)

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
