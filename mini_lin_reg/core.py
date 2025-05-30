import numpy as np
from .loss_functions import LossFunc
from .regularizers import Regularizer, NullRegularizer
from .optimizers import Optimizer

class MiniLinReg:
    """
    MiniLinReg
    -----
    Mini Linear Regression Model
    """
    def __init__(self,
                 input_size: int,
                 loss_function: LossFunc,
                 optimizer: Optimizer,
                 regularizer: Regularizer = None):
        # ===== input validation START =====
        if input_size == None:
            raise ValueError("input_size is required.")
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        
        if loss_function == None:
            raise ValueError("loss_function is required.")
        
        if optimizer == None:
            raise ValueError("optimizer is required.")
        # ===== input validation END =====

        self.input_size = input_size
        self.loss_function = loss_function
        self.regularizer = regularizer if regularizer else NullRegularizer()
        self.optimizer = optimizer

        # matrix dimensions
        # W shape: (input_size, 1)
        # B shape: (1, 1)
        self.weights = np.random.randn(input_size, 1) * 0.01
        self.bias    = np.random.randn(1, 1) * 0.01
    
    def predict(self,
                input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the model to generate predictions.

        Parameters
        -----
        input : np.ndarray of shape (N, input_size)
            The input feature matrix, where N is the number of samples.

        Return
        -----
        output : np.ndarray of shape (N, 1)
            The predicted values for each input sample.
        """

        # EQ    : output = input           @ W               + B
        # shape : (N, 1) = (N, input_size) @ (input_size, 1) + (N, 1)
        output = np.matmul(input, self.weights) + self.bias
        return output

    def train(self,
              input_train: np.ndarray,
              output_train: np.ndarray,
              input_test: np.ndarray | None = None,
              output_test: np.ndarray | None = None,
              batch_size: int = 1,
              epoch: int = 10) -> tuple[list, list]:
        """
        Trains the model to fit on the given dataset.

        Parameters
        ----------
        input : np.ndarray of shape (N, input_size)
            Input feature matrix, where N is the number of samples.

        output : np.ndarray of shape (N, 1)
            Target values corresponding to each input sample.

        batch_size : int
            Batch size used for training.
            - `1` performs stochastic gradient descent.
            - `1 < batch_size < N` performs mini-batch gradient descent.
            - `N` performs full-batch gradient descent.
            
            Default is `1`.

        epoch : int
            Number of epochs to train the model. \\
            Default is `10`.
        """

        _has_test_data: bool
        if input_test is not None and output_test is not None:
            _has_test_data = True
        elif input_test is None and output_test is None:
            _has_test_data = False
        else:
            raise ValueError("Both input_test and output_test must be provided.")
        
        # matrix dimensions
        # input  (X) shape: (N, input_size)
        # output (Y) shape: (N, 1)

        loss_arr_train = []
        loss_arr_test = []
        total_samples = input_train.shape[0]

        for _ in range(epoch):

            randomized_indices = np.random.permutation(total_samples)
            current_index = 0

            while current_index < total_samples:

                end_index = min(current_index + batch_size, total_samples)
                indices = randomized_indices[current_index: end_index]
                current_index += batch_size

                effective_batch_size = len(indices)

                # i shape: (b_size, input_size)
                # o shape: (b_size, 1)
                i: np.ndarray = input_train[indices]
                o: np.ndarray = output_train[indices]

                # ===== forward pass =====
                # EQ    : Y           = X                    @ W               + B
                # shape : (b_size, 1) = (b_size, input_size) @ (input_size, 1) + (b_size, 1)
                # NOTE: B is (1, 1) but numpy auto-broadcast it to (b_size, 1)
                y_pred = np.matmul(i, self.weights) + self.bias

                # ====== calculate loss =====
                # Loss is calculated by loss_function
                # EQ    : Loss        = loss(Y_true, Y_pred)
                # shape : (b_size, 1) = (b_size, 1)
                # loss here is not really necessary for now
                # loss = self.loss_function.calc_loss(y_true=o, y_pred=y_pred)

                # ===== backward pass (back propagation) =====
                # grad_Y is dLoss/dY (or dLoss/dY_pred)
                # EQ    : dLoss/dY    = loss'(Y_true, Y_pred)
                # shape : (b_size, 1) = (b_size, 1)
                grad_Y = self.loss_function.calc_grad(y_true=o, y_pred=y_pred)

                # grad_W is gradient of Loss w.r.t. weights
                # EQ    : dLoss/W         = X.T                  @ dLoss/dY
                # shape : (input_size, 1) = (input_size, b_size) @ (b_size, 1)
                grad_W = np.matmul(i.T, grad_Y) / effective_batch_size
                grad_W += self.regularizer.calc_grad(self.weights)

                # grad_B is gradient of Loss w.r.t. bias
                # EQ    : dLoss/B = avg_sum[dLoss/Y]
                # shape : (1, 1)  = avg_sum[(b_size, 1)]
                # Bias term should NEVER be regularized
                grad_B = np.sum(grad_Y, axis=0, keepdims=True) / effective_batch_size

                self.optimizer.step(self.weights, grad_W)
                self.optimizer.step(self.bias,    grad_B)

            # loss train calculation
            y_pred = self.predict(input_train)
            loss_train = self.loss_function.calc_loss(y_true=output_train, y_pred=y_pred)
            loss_arr_train.append(np.mean(loss_train))
            loss_test_msg = ""

            if _has_test_data:
                y_pred = self.predict(input_test)
                loss_test = self.loss_function.calc_loss(y_true=output_test, y_pred=y_pred)
                loss_arr_test.append(np.mean(loss_test))
                loss_test_msg = f" | test: {np.mean(loss_test):>10.4f}"

            print(f"Epoch: [{_+1:>3}/{epoch}] | {self.loss_function.__class__.__name__} | " +
                  f"train: {np.mean(loss_train):>10.4f}{loss_test_msg}")

        return loss_arr_train, loss_arr_test