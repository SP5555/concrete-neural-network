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

print("===== ===== ===== ===== =====")

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

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
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

# Save the plot to outputs folder
# plt.savefig("outputs/model3_loss_plot.png")
# plt.show()

# Predict on test set
y_train_pred = model3.predict(X_train)
y_test_pred = model3.predict(X_test)

# Calculate MSE
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Final Train MSE: {train_mse:.4f}")
print(f"Final Test MSE : {test_mse:.4f}")
