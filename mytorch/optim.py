from __future__ import annotations

import numpy as np
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


class Adam:
    """
    Implements Adam optimizer (Adaptive Moment Estimation).
    
    Adam combines the advantages of two other extensions of stochastic gradient descent:
    - AdaGrad: which works well with sparse gradients
    - RMSProp: which works well in on-line and non-stationary settings
    
    Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
    """
    params: List[Tensor]
    lr: float
    betas: tuple[float, float]
    eps: float
    m: List[np.ndarray]  # First moment vector (mean of gradients)
    v: List[np.ndarray]  # Second moment vector (uncentered variance of gradients)
    t: int  # Time step
    
    def __init__(
        self, 
        params: Iterable[Tensor], 
        lr: float = 0.001, 
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8
    ) -> None:
        """
        Args:
            params: An iterable of Tensors to optimize.
            lr: Learning rate (default: 0.001).
            betas: Coefficients used for computing running averages of gradient
                   and its square (default: (0.9, 0.999)).
            eps: Term added to the denominator to improve numerical stability
                 (default: 1e-8).
        """
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        # Initialize moment vectors
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
    
    def step(self) -> None:
        """Performs a single optimization step."""
        self.t += 1
        beta1, beta2 = self.betas
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad
            
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self) -> None:
        """Clears the gradients of all parameters this optimizer is managing."""
        for p in self.params:
            p.zero_grad()