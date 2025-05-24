from abc import ABC, abstractmethod
import numpy as np

class LossFunc(ABC):
    """
    LossFunc
    -----
    Abstract base class for all loss function implementations.

    All calculations must be done element wise
    """
    def __init__(self):
        pass

    @abstractmethod
    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the loss
        """
        pass

    @abstractmethod
    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the loss with respect to `y_pred`.
        """
        pass

class MAE(LossFunc):
    """
    Mean Absolute Error
    """
    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.abs(y_pred - y_true)

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true)

class MSE(LossFunc):
    """
    Mean Squared Error
    """
    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (0.5 * (y_pred - y_true) ** 2)

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true)