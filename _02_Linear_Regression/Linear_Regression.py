# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

try:
    from sklearn.neural_network import MLPRegressor
except ImportError as e:
    os.system("sudo pip3 install scikit-learn")
    from sklearn.neural_network import MLPRegressor


def ridge(data):
    X, y = read_data()
    m=X.shape[0]
    n=X.shape[1]
    Lambda=-0.1
    weight = np.matmul(np.linalg.inv(np.matmul(X.T, X)+Lambda*np.identity(n)), np.matmul(X.T, y))
    y_pred=weight @ data
    return weight @ data

def lasso(data):
    x, y = read_data()
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=10000, random_state=42)
    Data = data.reshape(1, -1)
    model.fit(x, y)
    y_pred = model.predict(Data)
    return y_pred

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
