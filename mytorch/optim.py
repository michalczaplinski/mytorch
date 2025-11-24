from __future__ import annotations

from typing import Iterable
import numpy as np
from .autograd import Tensor

class SGD:
    """
    Implements stochastic gradient descent.
    """
    params: list[Tensor]
    lr: float
    
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01) -> None:
        """
        Args:
            params: An iterable of Tensors to optimize.
            lr: Learning rate.
        """
        self.params = list(params)
        self.lr = lr

    def step(self) -> None:
        """Performs a single optimization step."""
        for p in self.params:
            if p.grad is not None:
                # The core update rule
                p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        """
        Clears the gradients of all parameters this optimizer is managing.
        (Convenience method, same as model.zero_grad()).
        """
        for p in self.params:
            p.zero_grad()


class Adam:
    """
    Implements Adam algorithm.
    """
    params: list[Tensor]
    lr: float
    beta1: float
    beta2: float
    eps: float
    state: dict[Tensor, dict[str, np.ndarray]]
    t: int
    
    def __init__(self, params: Iterable[Tensor], lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {}
        self.t = 0 # Time step (needed for bias correction)

    def step(self) -> None:
        self.t += 1
        
        for p in self.params:
            if p.grad is None:
                continue

            # Initialize state for this parameter if it doesn't exist
            if p not in self.state:
                self.state[p] = {
                    'm': np.zeros_like(p.data), 
                    'v': np.zeros_like(p.data)
                }
            
            state = self.state[p]
            m, v = state['m'], state['v']
            g = p.grad

            # 1. Update raw moving averages (Momentum & RMSProp)
            # Note: We use in-place operations ([:]) to update the numpy arrays inside the dict
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)

            # 2. Bias Correction
            # (Boosts magnitude in early steps to counter initialization at zero)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # 3. Update Parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Clears the gradients of all parameters."""
        for p in self.params:
            p.zero_grad()