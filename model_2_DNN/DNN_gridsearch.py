import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor

from model_creator import create_model

# ===== Load Dataset =====
concrete_data = pd.read_csv("data/Concrete_Data_Yeh.csv")
X = concrete_data.iloc[:, :-1].values
y = concrete_data.iloc[:, -1].values

# ===== Standardize =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("===== ===== ===== ===== =====")

# Define safe parameter grid
param_grid = {
    'num_layers': [2, 3],
    'num_neurons': [64],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.0, 0.2, 0.4],  # avoid high dropout
    'momentum': [0.5, 0.7, 0.9]
}

# Create KerasRegressor
model = KerasRegressor(
    model=create_model,
    epochs=40,
    batch_size=32,
    verbose=0,
    **param_grid
)

# Run GridSearchCV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold,
    scoring='neg_mean_squared_error',
    verbose=3
)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# grid_result = grid_search.fit(X_scaled, y)
grid_result = grid_search.fit(X_train, y_train)

FORMAT_WIDTH = 9
# Save grid search results to file
with open("outputs/GS_best_comp.txt", "w") as f:
    # Best result
    f.write(f"Best MSE: {abs(grid_result.best_score_):>{FORMAT_WIDTH}.4f} | Using {grid_result.best_params_}\n\n")

    # Detailed results for all parameter combinations
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        f.write(f"MSE: {abs(mean):>{FORMAT_WIDTH}.4f} | STD: {std:>{FORMAT_WIDTH}.4f} | Using: {param}\n")