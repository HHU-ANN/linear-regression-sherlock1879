# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

try:
    from sklearn.linear_model import RANSACRegressor
except ImportError as e:
    os.system("sudo pip3 install scikit-learn")
    from sklearn.linear_model import RANSACRegressor

try:
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    os.system("sudo pip3 install scikit-learn")
    from sklearn.linear_model import LinearRegression

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
    model = LinearRegression()
    ransac = RANSACRegressor(base_estimator=model, min_samples=50, residual_threshold=3, max_trials=200,stop_n_inliers=108,stop_probability=0.96)
    Data = data.reshape(1, -1)
    ransac.fit(x, y)
    y_pred = ransac.predict(Data)
    return y_pred

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
