import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y, learning_rate=0.001, epochs=100, batch_size=None, verbose=True):
        self.theta = np.random.randn(X.shape[1])
        m = X.shape[0]

        for epoch in range(epochs):
            num_batches = m // batch_size if batch_size else 1
            for batch in range(num_batches):
                if batch_size:
                    indices = np.random.choice(m, batch_size)
                    X_batch = X[indices]
                    y_batch = y[indices]
                else:
                    X_batch = X
                    y_batch = y

                y_pred = X_batch.dot(self.theta)
                err = y_pred - y_batch
                grad = 2 * X_batch.T.dot(err)
                avg_grad = grad / X_batch.shape[0]
                self.theta -= learning_rate * avg_grad
    
    def predict(self, X):
        return X.dot(self.theta)

def pol_reg(degree, X, y, fold_count, test_size=0.3, random_state=23):
    kf = KFold(fold_count)

    pfeat = PolynomialFeatures(degree)
    X_p = pfeat.fit_transform(X)

    # standarize to avoid overflow
    scaler = StandardScaler()
    X_p = scaler.fit_transform(X_p)

    error_history = []
    models = []
    splits = [] # keep track of how the data was split each fold so we can reuse it for MSE calculation later
    fold_measures = []

    for train_index, test_index in kf.split(X_p):
        X_train, X_test = X_p[train_index], X_p[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        pred_avg = np.mean(y_pred)
        var = np.mean(np.square(y_pred - pred_avg))
        bias2 = np.mean(np.square(y_test - pred_avg))
        total_err = var + bias2

        fold_measures.append((var, bias2, total_err))

        error_history.append(total_err)
        models.append(model)
        splits.append((X_train, X_test, y_train, y_test))

    min_err_idx = np.argmin(error_history)
    best_model = models[min_err_idx]
    best_split = splits[min_err_idx]
    return best_model, best_split, fold_measures

dataset = pd.read_csv("../data/Concrete_Data_Yeh.csv")
X = dataset.drop("csMPa", axis=1).values

# assign values for y from dataset
y = dataset["csMPa"].values

# degrees 1-5
for degree in range(1,6):
    # fold measures to measure variance and bias later, need to add still
    best_model, best_split, fold_measures = pol_reg(degree, X, y, 5)

    X_train, X_test, y_train, y_test = best_split

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("Degree:", degree, )
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    print("========")

