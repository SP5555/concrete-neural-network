from abc import ABC, abstractmethod
import numpy as np

class Regularizer(ABC):
    """
    Regularizer
    -----
    Abstract base class for all loss function implementations.

    All calculations must be done element wise
    """
    def __init__(self):
        pass

    @abstractmethod
    def calc_penalty(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the penalty for given weights
        """
        pass

    @abstractmethod
    def calc_grad(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the regularized term with respect to weights.
        """
        pass

class NullRegularizer(Regularizer):
    """
    No Regularization
    """
    def calc_penalty(self, weights: np.ndarray) -> np.ndarray:
        penalty = np.zeros_like(weights)
        return penalty
    
    def calc_grad(self, weights: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(weights)
        return grad

class L1Regularizer(Regularizer):
    """
    L1 Regularization
    """
    def calc_penalty(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        return penalty
    
    def calc_grad(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        return grad

class L2Regularizer(Regularizer):
    """
    L2 Regularization
    """
    def calc_penalty(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        return penalty
    
    def calc_grad(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        return grad