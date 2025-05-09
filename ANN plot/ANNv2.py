import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras import Input
import pandas as pd
import os
import matplotlib.pyplot as plt

# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx('float32')

# Load dataset from local file
df = pd.read_csv('Concrete_Data_Yeh.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Check for NaNs and fix if needed
print("NaNs in X:", np.isnan(X).sum())
print("NaNs in y:", np.isnan(y).sum())
X = np.nan_to_num(X)
y = np.nan_to_num(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model creation function
def create_model(num_layers=1, num_neurons=64, activation='relu', dropout_rate=0.0, momentum=0.9):
    model = Sequential()
    model.add(Input(shape=(X_scaled.shape[1],)))
    model.add(Dense(num_neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Regression output
    optimizer = SGD(learning_rate=0.01, momentum=momentum, clipvalue=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model

# Define safe parameter grid
param_grid = {
    'num_layers': [2, 3],
    'num_neurons': [32, 64],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.0, 0.2],  # avoid high dropout
    'momentum': [0.0, 0.9] # changed momentum
}

# Create KerasRegressor
model = KerasRegressor(
    model=create_model,
    epochs=50,
    batch_size=32,
    verbose=0,
    num_layers=3,
    num_neurons=64,
    activation='relu',
    dropout_rate=0.0,
    momentum=0.9
)

# Run GridSearchCV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error', verbose=1) # set verbose = 1 to show progress per epoch
print("Running GridSearchCV... this may take several minutes.") # debug statement to make sure it's running
grid_result = grid_search.fit(X_scaled, y)

# Print best result
print("Best MSE: %f using %s" % (abs(grid_result.best_score_), grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print("MSE: %f (±%f) with: %r" % (abs(mean), std, param))

# Re-train best model based on best parameters
best_params = grid_result.best_params_ # use .best_params_ to give us the best parameters based on best results
best_model = KerasRegressor(
    model=create_model,
    epochs=50,
    batch_size=32,
    verbose=0, 
    num_layers=best_params['num_layers'],
    num_neurons=best_params['num_neurons'],
    activation=best_params['activation'],
    dropout_rate=best_params['dropout_rate'],
    momentum=best_params['momentum']
)

# Fit and store the training history
history = best_model.fit(X_scaled, y, validation_split=0.2)

plt.figure(figsize=(10, 6))
plt.plot(history.history_['loss'], label='Train MSE')
plt.plot(history.history_['val_loss'], label='Test MSE')
plt.title('MSE Loss Over Epochs (Best Model)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot in the current directory
plt.savefig("best_model_loss_plot.png")

# Show the plot
plt.show()

