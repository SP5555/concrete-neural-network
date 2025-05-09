from .models import TFDNN

class TFModelWrapper:
    def __init__(self,
                 model: TFDNN,
                 loss: str,
                 optimizer: str,
                 metrics: list[str]):
        # Build and compile the model in one step
        self.model: TFDNN = model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self,
              x_train,
              y_train,
              epochs,
              batch_size,
              validation_data=None,
              callbacks=None,
              verbose=True):
        return self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()
