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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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






# best_params = {
#   'num_layers': 2, 
#   'neurons_layer2': 8, 
#   'neurons_layer1': 32, 
#   'learning_rate': 0.003, 
#   'dropout_rate': 0.1
# }

best_params = {
  'num_layers': 2,
  'neurons_layer2': 16, 
  'neurons_layer1': 64, 
  'learning_rate': 0.001, 
  'dropout_rate': 0.3
}



#Create the model based on the best hyperparameters
model = Sequential()
model.add(Dense(best_params['neurons_layer1'], activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dropout(best_params['dropout_rate']))

model.add(Dense(best_params['neurons_layer2'], activation='relu'))
model.add(Dropout(best_params['dropout_rate']))

model.add(Dense(1))  # Output layer for regression

optimizer = SGD(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])



#Train this model on the training data.
history = model.fit(train_data, train_labels, epochs=300, batch_size=32, verbose=1)

# Plot loss vs. epochs
plt.plot(history.history['loss'])
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.show()

#Use this model on a test set. GENERATE PREDICTIONS. 
predictions = model.predict(test_data).squeeze()

#Evaluate model. MSE. 
mse = mean_squared_error(test_labels, predictions)
print(f"\nTest MSE: {mse:.2f}")

# If Mean Squared Error = 195.25
# Then the actual number is + or - 13.97 MPA of the prediction.
print("Minimum target value in our dataset:", np.min(y))
print("Maximum target value in our dataset:", np.max(y))
print("The prediction is + or - ", np.sqrt(mse), "MPA of the actual value")