from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


#### SECOND MODEL (DNN) START########

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasRegressor

from model_creator import create_model

# ===== Load Dataset =====
concrete_data = pd.read_csv("data/Concrete_Data_Yeh.csv")
X = concrete_data.iloc[:, :-1].values
y = concrete_data.iloc[:, -1].values

# Define the best-known hyperparameters manually
best_params = {
    'activation': 'tanh',
    'dropout_rate': 0.0,
    'momentum': 0.7,
    'num_layers': 2,
    'num_neurons': 64
}

# Build the model using the best params
best_model = KerasRegressor(
    model=create_model,
    epochs=50,
    batch_size=32,
    verbose=3,
    **best_params
)

# Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# get original data
X_test_8 = X_test
y_test_8 = y_test

# ===== Standardize =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test))

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history_['loss'], label='Train MSE')
plt.plot(history.history_['val_loss'], label='Test MSE')
plt.title('MSE Loss Over Epochs (Best Model)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

#### SECOND MODEL (DNN) START #######


#### THIRD MODEL (DNN) START #############

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasRegressor

from model_creator import create_model3

# ===== Load Dataset =====
concrete_data = pd.read_csv("data/Concrete_Data_Yeh.csv")
X = concrete_data.iloc[:, :-1].values
y = concrete_data.iloc[:, -1].values

# ===== Standardize =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the best-known hyperparameters manually
model3_params = {
    'dropout_rate': 0.0,
    'momentum': 0.7,
}

# Build the model using the best params
model3 = KerasRegressor(
    model=create_model3,
    epochs=50,
    batch_size=32,
    verbose=3,
    **model3_params
)

history = model3.fit(X_train, y_train, validation_data=(X_test, y_test))

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history_['loss'], label='Train MSE')
plt.plot(history.history_['val_loss'], label='Test MSE')
plt.title('MSE Loss Over Epochs (Best Model)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

#### THIRD MODEL (DNN) END #############


####### FIRST MODEL POLYNOMIAL REGRESSION ###########

import sys
import os
import time
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from mini_lin_reg import MiniLinReg
from mini_lin_reg.loss_functions import MAE, MSE
from mini_lin_reg.optimizers import SGD, Momentum, Adam
from mini_lin_reg.regularizers import L1Regularizer, L2Regularizer

from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("data/Concrete_Data_Yeh.csv")
X = dataset.drop("csMPa", axis=1).values
y = dataset["csMPa"].values.reshape(-1, 1)

deg2 = PolynomialFeatures(1)
X_poly = deg2.fit_transform(X)

X_poly = StandardScaler().fit_transform(X_poly)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2)

model = MiniLinReg(
    input_size=X_poly.shape[1],
    loss_function=MSE(),
    optimizer=Momentum(learn_rate=0.002),
    regularizer=L2Regularizer(reg_lambda=0.02)
)

t1 = time.time()
loss_train, loss_test = model.train(
    input_train=X_train_poly,
    output_train=y_train_poly,
    input_test=X_test_poly,
    output_test=y_test_poly,
    batch_size=32,
    epoch=50
)
t2 = time.time()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(loss_train, label='Train MSE')
plt.plot(loss_test, label='Test MSE')
plt.title('MSE Loss Over Epochs (MiniLinReg Model)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

####### FIRST MODEL POLYNOMIAL REGRESSION END ###########


@app.get("/", response_class=HTMLResponse)
def greet_json():
    y_pred_dnn1 = best_model.predict(X_test)
    train_mse = mean_squared_error(y_train, best_model.predict(X_train))
    test_mse = mean_squared_error(y_test, y_pred_dnn1)
    best_model_mse = f"{test_mse:.4f}"
    print(f"Final Test MSE (second DNN): {test_mse:.4f}")

    y_pred_dnn2 = model3.predict(X_test)
    print(f"Final Test MSE (third model DNN) : {mean_squared_error(y_test, y_pred_dnn2):.4f}")

    y_pred_poly_reg = model.predict(X_test_poly).flatten()
    print(f"Test MSE (poly reg): {mean_squared_error(y_test_poly, y_pred_poly_reg):.4f}")

    feature_columns = [
        "cement (kg/m³)",
        "slag (kg/m³)",
        "ash (kg/m³)",
        "water (kg/m³)",
        "superplasticizer (kg/m³)",
        "coarseaggregate (kg/m³)",
        "fineaggregate (kg/m³)",
        "age (days)"
    ]
    target_column = "csMPa (MPa)"

    df_X_test = pd.DataFrame(X_test_8, columns=feature_columns)
    df_y_test = pd.DataFrame(y_test_8, columns=[target_column])

    df_preds = pd.DataFrame({
        "bst_mdl_pred (MPa)":   y_pred_dnn1,
        "pred_dnn2 (MPa)":   y_pred_dnn2,
        "y_pred_poly_reg (MPa)":   y_pred_poly_reg,
    })

    df_test = pd.concat([df_X_test, df_y_test, df_preds], axis=1)
    df_test_html = df_test.to_html()

    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <title>FastAPI Greeting</title>
      </head>
      <body>
        <h1>best model mse: {best_model_mse}</h1>
        <div>{df_test_html}</div>
      </body>
    </html>
    """
