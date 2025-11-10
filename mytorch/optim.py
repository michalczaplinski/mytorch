from __future__ import annotations

from typing import Iterable, List
from .autograd import Tensor

class SGD:
    """
    Implements stochastic gradient descent.
    """
    params: List[Tensor]
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