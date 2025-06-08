from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Input

# Define model creation function
def create_model(num_layers=1, num_neurons=64, activation='relu', dropout_rate=0.0, momentum=0.9):
    model = Sequential()
    model.add(Input(shape=(8,))) # !!! hardcoded input size !!!

    model.add(Dense(num_neurons, activation=activation))
    model.add(Dropout(dropout_rate))

    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # Regression output

    optimizer = SGD(learning_rate=0.01, momentum=momentum, clipvalue=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model

# Define model creation function
def create_model3(dropout_rate=0.0, momentum=0.9):
    model = Sequential([
        Input(shape=(8,)), # !!! hardcoded input size !!!
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-2, l2=1e-2)),
        Dropout(dropout_rate),
        Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-2, l2=1e-2)),
        Dropout(dropout_rate),
        Dense(8, activation='relu', kernel_regularizer=l1_l2(l1=1e-2, l2=1e-2)),
        Dense(1)
    ])
    optimizer = SGD(learning_rate=0.01, momentum=momentum, clipvalue=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model