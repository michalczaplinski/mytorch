# nn.py

import numpy as np
from .autograd import Tensor, Module
from .autograd import NLLLoss as NLLLossFn # Import the Function

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