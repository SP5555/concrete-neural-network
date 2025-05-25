from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Optimizer
    -----
    Abstract base class for all optimizer implementations.

    Optimizers should update the weights in-place.
    """
    def __init__(self, learn_rate: float):
        
        if learn_rate <= 0.0: 
            raise ValueError("learn_rate must be positive.")
        if learn_rate >= 1.0:
            print("learn_rate is too high, consider keeping it low.")
        
        self.lr = learn_rate
        pass

    @abstractmethod
    def step(self, weights: np.ndarray, grad: np.ndarray) -> None:
        pass

class SGD(Optimizer):
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

    def step(self, weights: np.ndarray, grad: np.ndarray) -> None:
        weights -= self.lr * grad

class Momentum(Optimizer):
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

    def step(self, weights: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Adam(Optimizer):
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

    def step(self, weights: np.ndarray, grad: np.ndarray) -> None:
        raise NotImplementedError

