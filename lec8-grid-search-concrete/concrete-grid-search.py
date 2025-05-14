# USE PYTHON 3.10.13

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras.optimizers import SGD
from ucimlrepo import fetch_ucirepo 
import numpy as np
  
# fetch dataset 
concrete_compressive_strength = fetch_ucirepo(id=165) 
  
# data (as pandas dataframes) 
X = concrete_compressive_strength.data.features 
y = concrete_compressive_strength.data.targets 
  
# metadata 
#print(concrete_compressive_strength.metadata) 
  
# variable information 
#print(concrete_compressive_strength.variables) 


# Preprocess the data by scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape of X", X.shape)
y = np.squeeze(y)
print("Shape of y", y.shape, '\n')


# Split the data into training and testing sets
train_size = int(0.8 * len(X_scaled))
train_data, train_labels = X_scaled[:train_size], y[:train_size]
test_data, test_labels = X_scaled[train_size:], y[train_size:]

print("printing train_data[0]", train_data[0], '\n')


# Define the neural network architecture
def create_model(num_layers=2, learning_rate=0.001, dropout_rate=0.2, neurons_layer1=64, neurons_layer2=32):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(dropout_rate))
    for i in range(num_layers - 1):
        model.add(Dense(neurons_layer2, activation='relu'))
        model.add(Dropout(dropout_rate))
    #model.add(Dense(1, activation='sigmoid')) remove this because we are doing regression. not classification.
    model.add(Dense(1))

    optimizer = SGD(learning_rate = learning_rate)
    #model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse']) # could also use mae
    return model


# Define the hyperparameters to tune using grid search
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.003, 0.01],
    'dropout_rate': [0, 0.1, 0.2, 0.3, 0.4],
    'num_layers': [1, 2],
    'neurons_layer1': [4, 8, 16, 32, 64, 128],
    'neurons_layer2': [4, 8, 16, 32, 64, 128]
}


# Create the grid search object
model = KerasRegressor(build_fn=create_model, epochs=50, batch_size=32, verbose=0)
# 3 is the number of folds in k-fold cross validation. one fold for validation and k-1 folds used for training
# for each combination of parameters, the model is trained on the training data and tested o the validation data.
# the average performance of the model (with the selected combination of parameters) over k-folds is used as the estimate for the model's perfromance for tha particular combination of hyperparameters. 

#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')

# Train the model with grid search
grid_result = grid.fit(train_data, train_labels)

# Print the best hyperparameters and accuracy
print("Best parameters: ", grid_result.best_params_)
print("Best MSE: {:.2f}".format(-grid_result.best_score_))



# Test this model on the test set. see the bestANN.py file
