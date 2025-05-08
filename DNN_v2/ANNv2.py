import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Input
import os

# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx('float32')

# Load dataset
concrete_data = fetch_ucirepo(id=165)
X = concrete_data.data.features
y = concrete_data.data.targets.values.ravel()

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
    'momentum': [0.5, 0.9]
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
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid_result = grid_search.fit(X_scaled, y)

# Print best result
print("Best MSE: %f using %s" % (abs(grid_result.best_score_), grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print("MSE: %f (±%f) with: %r" % (abs(mean), std, param))
