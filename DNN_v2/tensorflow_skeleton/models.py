import tensorflow as tf

# Base Class
class TFDNN:
    def __init__(self,
                 input_size: int = None):
        if input_size == None:
            raise KeyError("input_size can't be empty.")
        if input_size == 0:
            raise KeyError("input_size can't be 0.")

        self.input_size: int = input_size
        self.model: TFDNN    = None

    def get_model(self):
        return self.model

# Make different models as you wish
class DNN_i_12_12_1_Relu(TFDNN):
    def __init__(self, input_size = None):
        super().__init__(input_size)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_size,)),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])

class DNN_model2(TFDNN):
    def __init__(self, input_size = None):
        super().__init__(input_size)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_size,)),
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # Output layer for regression
        ])
