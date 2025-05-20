import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

def pol_reg(degree, X, y, fold_count, test_size=0.3, random_state=33):
    kf = KFold(fold_count)

    pfeat = PolynomialFeatures(degree)
    X_p = pfeat.fit_transform(X)

    # standarize to avoid overflow
    scaler = StandardScaler()
    X_p = scaler.fit_transform(X_p)

    print(np.shape(X_p))

    error_history = []
    models = []
    splits = [] # keep track of how the data was split each fold so we can reuse it for MSE calculation later
    fold_measures = []

    for train_index, test_index in kf.split(X_p):
        X_train, X_test = X_p[train_index], X_p[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Lasso(alpha = 0.0001, tol = 0.01, max_iter=1000)
        # model = Lasso(alpha = 0.0015, tol = 0.001, max_iter=100000)
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
    best_model, best_split, fold_measures = pol_reg(degree, X, y, 10)

    X_train, X_test, y_train, y_test = best_split

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("Degree:", degree, )
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    print("========")

