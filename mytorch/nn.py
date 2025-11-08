# nn.py

import numpy as np
from .autograd import Tensor
from .autograd import NLLLoss as NLLLossFn # Import the Function

class Module:
    """
    Base class for all neural network modules (layers and models).
    """
    
    def __call__(self, *args, **kwargs):
        """Makes the module callable, e.g., model(x)"""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """(Subclass must implement) Defines the forward pass."""
        raise NotImplementedError

    def parameters(self):
        """
        Returns a generator of all parameters (Tensors with requires_grad=True)
        in this module and its sub-modules.
        """
        for name, attr in self.__dict__.items():
            if isinstance(attr, Tensor) and attr.requires_grad:
                yield attr
            elif isinstance(attr, Module):
                yield from attr.parameters() # Recurse

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for p in self.parameters():
            p.zero_grad()

class Linear(Module):
    """
    A simple linear layer: y = x @ W + b
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # Kaiming He initialization
        stdv = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.normal(0, stdv, (in_features, out_features)), 
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        return (x @ self.weight) + self.bias

class ReLU(Module):
    """A simple stateless ReLU activation module."""
    def forward(self, x):
        return x.relu()

class LogSoftmax(Module):
    """A simple stateless LogSoftmax module."""
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
        
    def forward(self, x):
        return x.log_softmax(axis=self.axis)

# --- Loss Function ---

def nll_loss(log_probs, targets):
    """
    Computes the Negative Log Likelihood loss.
    (This is a functional wrapper for the NLLLoss Function)
    """
    # Pass targets as a non-Tensor kwarg so only Tensors are tracked in the graph
    return NLLLossFn.apply(log_probs, targets=targets)