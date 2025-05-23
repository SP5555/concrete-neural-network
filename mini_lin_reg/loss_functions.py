from abc import ABC, abstractmethod
import numpy as np

class LossFunc(ABC):
    """
    LossFunc
    -----
    Abstract base class for all loss function implementations.

    All calculations should work element wise
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

class MSE(LossFunc):
    """
    Mean Squared Error
    """
    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        return loss

    def calc_grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        return grad