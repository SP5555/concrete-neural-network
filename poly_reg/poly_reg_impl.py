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

import matplotlib.pyplot as plt

from mini_lin_reg import MiniLinReg
from mini_lin_reg.loss_functions import MAE, MSE
from mini_lin_reg.optimizers import SGD, Adam

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
    optimizer=Adam(learn_rate=0.02),
)
loss_name = "MSE"

t1 = time.time()
loss_train, loss_test = model.train(
    input_train  =X_train,
    output_train =y_train,
    input_test   =X_test,
    output_test  =y_test,
    batch_size   =16,
    epoch        =60
)
t2 = time.time()
print(f"Training time taken: {(t2 - t1) * 1000:.4f} ms")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE (from sklearn): {mse:.4f}")

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(loss_train, label=f'Train {loss_name}')
plt.plot(loss_test, label=f'Test {loss_name}')
plt.title(f'{loss_name} Loss Over Epochs (MiniLinReg Model)')
plt.xlabel('Epoch')
plt.ylabel(f'{loss_name} Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to outputs folder
# plt.savefig("outputs/poly_reg_loss_plot.png")
plt.show()