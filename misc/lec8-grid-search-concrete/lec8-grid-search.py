# USE PYTHON 3.10.13

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD

boston = load_boston()
# Preprocess the data by scaling the features
scaler = StandardScaler()
X = scaler.fit_transform(boston.data)
y = boston.target

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
train_data, train_labels = X[:train_size], y[:train_size]
test_data, test_labels = X[train_size:], y[train_size:]

print(train_data[0])

# Define the neural network architecture
def create_model(num_layers=2, learning_rate=0.001, dropout_rate=0.2, neurons_layer1=64, neurons_layer2=32):
    model = Sequential()
    model.add(Dense(neurons_layer1, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(dropout_rate))
    for i in range(num_layers - 1):
        model.add(Dense(neurons_layer2, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = SGD()
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model

# Define the hyperparameters to tune using grid search
param_grid = {
    'learning_rate': [0.001, 0.01],
    'dropout_rate': [0.2, 0.3],
    'num_layers': [2, 3],
    'neurons_layer1': [64, 128],
    'neurons_layer2': [32, 64]
}

# Create the grid search object
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)
# 3 is the number of folds in k-fold cross validation. one fold for validation and k-1 folds used for training
# for each combination of parameters, the model is trained on the training data and tested o the validation data.
# the average performance of the model (with the selected combination of parameters) over k-folds is used as the estimate for the model's perfromance for tha particular combination of hyperparameters. 

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Train the model with grid search
grid_result = grid.fit(train_data, train_labels)

# Print the best hyperparameters and accuracy
print("Best parameters: ", grid_result.best_params_)
print("Best accuracy: {:.2f}%".format(grid_result.best_score_ * 100))
