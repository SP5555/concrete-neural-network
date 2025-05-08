import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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

        error_history.append(total_err)
        models.append(model)

    min_err_idx = np.argmin(error_history)
    best_model = models[min_err_idx]
    return best_model

dataset = pd.read_csv("Concrete_Data_Yeh.csv")
X = dataset.drop("csMPa", axis=1).values

# assign values for y from dataset
y = dataset["csMPa"].values
best_model = pol_reg(5, X, y, 5)
