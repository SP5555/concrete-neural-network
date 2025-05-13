import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor

from model_creator import create_model

# ===== Backend and OS setup =====
# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.makedirs("outputs", exist_ok=True)
tf.keras.backend.set_floatx('float32')

# ===== Load Dataset =====
concrete_data = fetch_ucirepo(id=165)
X = concrete_data.data.features
y = concrete_data.data.targets.values.ravel()

# ===== Check and Fix NaNs =====
print("NaNs in X (input features)")
print(np.isnan(X).sum())
print("NaNs in y (output target)")
print(np.isnan(y).sum())
X = np.nan_to_num(X)
y = np.nan_to_num(y)

# ===== Standardize =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("===== ===== ===== ===== =====")

# Define the best-known hyperparameters manually
best_params = {
    'activation': 'tanh',
    'dropout_rate': 0.0,
    'momentum': 0.5,
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

# Train the model and save training history
history = best_model.fit(X_scaled, y, validation_split=0.2)

# Make sure output folder exists
os.makedirs("outputs", exist_ok=True)

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
plt.savefig("outputs/best_model_loss_plot.png")
plt.show()
