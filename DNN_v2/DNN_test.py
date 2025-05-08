import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import KerasClassifier

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model creation function
from tensorflow.keras import Input

def create_model(num_layers=1, num_neurons=64, activation='relu', dropout_rate=0.0, momentum=0.9):
    model = Sequential()
    model.add(Input(shape=(X_scaled.shape[1],)))  # ✅ Use Input layer
    model.add(Dense(num_neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    optimizer = SGD(learning_rate=0.01, momentum=momentum)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Define parameter grid for GridSearchCV
param_grid = {
    'num_layers': [3],
    'num_neurons': [32, 64],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.2, 0.5],
    'momentum': [0.5, 0.9]
}

# Create KerasClassifier with all model parameters explicitly defined
model = KerasClassifier(
    model=create_model,
    epochs=10,
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
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='accuracy')
grid_result = grid_search.fit(X_scaled, y)

# Print best result
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, std, param))
