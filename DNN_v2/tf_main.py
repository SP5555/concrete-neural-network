import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow_skeleton import config
from tensorflow_skeleton.utils import standardscalar_transform
from tensorflow_skeleton.models import DNN_i_12_12_1_Relu, DNN_model2
from tensorflow_skeleton.wrapper import TFModelWrapper

def main():

    # Data Processing
    dataset = pd.read_csv("Concrete_Data_Yeh.csv")
    X = dataset.drop("csMPa", axis=1).values
    y = dataset["csMPa"].values
    X_scaled = standardscalar_transform(X)
    x_train, x_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)

    # Initialize Model
    wrapper = TFModelWrapper(
        model=DNN_model2(input_size=x_train.shape[1]).get_model(),
        loss=config.LOSS,
        optimizer=config.OPTIMIZER,
        metrics=config.METRICS
    )

    # Train
    # wrapper.summary()
    wrapper.train(
        x_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=True # False to make it shut up
    )

    # Evaluate
    val_loss, val_mae = wrapper.evaluate(x_val, y_val)
    print(f"Validation Loss: {val_loss:.4f}, Validation MSE: {val_mae:.4f}")

if __name__ == "__main__":
    main()
