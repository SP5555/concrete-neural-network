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
    def __init__(self, learn_rate: float, beta: float = 0.9):
        super().__init__(learn_rate)
        if not 0.0 <= beta < 1.0:
            raise ValueError("Beta must be between [0.0, 1.0]")
        self.beta = beta
        self.m = {}

    def step(self, weights: np.ndarray, grad: np.ndarray) -> None:
        w_id = id(weights)
        if w_id not in self.m:
            self.m[w_id] = np.zeros_like(weights)
        
        self.m[w_id] = self.beta * self.m[w_id] + (1 - self.beta) * grad

        weights -= self.lr * self.m[w_id]

class Adam(Optimizer):
    def __init__(self, learn_rate: float, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__(learn_rate)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0 :
            raise ValueError("Both beta1 and beta2 must be between [0.0, 1.0).")
        if beta1 >= 0.95:
            print(f"Warning: beta1 = {beta1:.3f} may cause strong \"gliding\" behavior. " +
                    "Consider keeping it less than 0.95")

        self.beta1 = beta1
        self.beta2 = beta2
        # 1st moment
        self.m = {}
        # 2nd moment
        self.v = {}
        # step counter
        self.t = 0

    def step(self, weights: np.ndarray, grad: np.ndarray) -> None:
        self.t += 1

        w_id = id(weights)
        if w_id not in self.m:
            self.m[w_id] = np.zeros_like(weights)
            self.v[w_id] = np.zeros_like(weights)

        # update 1st and 2nd moments
        self.m[w_id] = self.beta1 * self.m[w_id] + (1-self.beta1) * grad
        self.v[w_id] = self.beta2 * self.v[w_id] + (1-self.beta2) * np.square(grad)

        # this is some sort of scaling, known as "bias-correction"
        m_hat = self.m[w_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[w_id] / (1 - self.beta2 ** self.t)

        weights += -1 * self.lr * m_hat / np.sqrt(v_hat + 1e-12)