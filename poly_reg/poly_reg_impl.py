# path fix
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from mini_lin_reg import MiniLinReg
from mini_lin_reg.loss_functions import MSE
from mini_lin_reg.optimizers import SGD

from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("data/Concrete_Data_Yeh.csv")
X = dataset.drop("csMPa", axis=1).values
y = dataset["csMPa"].values.reshape(-1, 1)

deg2 = PolynomialFeatures(1)
X = deg2.fit_transform(X)

X = StandardScaler().fit_transform(X)
print(f"Dataset shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MiniLinReg(
    input_size=X.shape[1],
    loss_function=MSE(),
    optimizer=SGD(learn_rate=0.003),
)

t1 = time.time()
model.train(X_train, y_train, batch_size=32, epoch=50)
t2 = time.time()
print(f"Training time taken: {(t2 - t1) * 1000:.4f} ms")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
