# nn.py

from __future__ import annotations

import numpy as np
import pickle
from pathlib import Path
from typing import Any, Generator, Optional, Dict, Union
from .autograd import Tensor, nll_loss

class Module:
    """
    Base class for all neural network modules (layers and models).
    """
    
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """Makes the module callable, e.g., model(x)"""
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """(Subclass must implement) Defines the forward pass."""
        raise NotImplementedError

    def parameters(self) -> Generator[Tensor, None, None]:
        """
        Returns a generator of all parameters (Tensors with requires_grad=True)
        in this module and its sub-modules.
        """
        for name, attr in self.__dict__.items():
            if isinstance(attr, Tensor) and attr.requires_grad:
                yield attr
            elif isinstance(attr, Module):
                yield from attr.parameters() # Recurse

    def zero_grad(self) -> None:
        """Sets gradients of all parameters to zero."""
        for p in self.parameters():
            p.zero_grad()
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing the state of the module.
        
        Returns:
            A dictionary mapping parameter names to their data arrays.
        """
        state = {}
        
        def add_to_state(module: Module, prefix: str = '') -> None:
            for name, attr in module.__dict__.items():
                if isinstance(attr, Tensor) and attr.requires_grad:
                    key = f"{prefix}{name}" if prefix else name
                    state[key] = attr.data.copy()
                elif isinstance(attr, Module):
                    submodule_prefix = f"{prefix}{name}." if prefix else f"{name}."
                    add_to_state(attr, submodule_prefix)
        
        add_to_state(self)
        return state
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Loads the module state from a state dictionary.
        
        Args:
            state_dict: A dictionary mapping parameter names to numpy arrays.
        """
        def load_from_state(module: Module, prefix: str = '') -> None:
            for name, attr in module.__dict__.items():
                if isinstance(attr, Tensor) and attr.requires_grad:
                    key = f"{prefix}{name}" if prefix else name
                    if key in state_dict:
                        attr.data = state_dict[key].copy()
                    else:
                        raise KeyError(f"Key '{key}' not found in state_dict")
                elif isinstance(attr, Module):
                    submodule_prefix = f"{prefix}{name}." if prefix else f"{name}."
                    load_from_state(attr, submodule_prefix)
        
        load_from_state(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the module's state to a file.
        
        Args:
            path: Path where the state will be saved.
        """
        path = Path(path)
        state = self.state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Loads the module's state from a file.
        
        Args:
            path: Path to the saved state file.
        """
        path = Path(path)
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.load_state_dict(state)

class Linear(Module):
    """
    A simple linear layer: y = x @ W + b
    """
    weight: Tensor
    bias: Tensor
    
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # Kaiming He initialization
        stdv = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.normal(0, stdv, (in_features, out_features)), 
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias

class ReLU(Module):
    """A simple stateless ReLU activation module."""
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Module):
    """A simple stateless Sigmoid activation module."""
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Tanh(Module):
    """A simple stateless Tanh activation module."""
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

class LogSoftmax(Module):
    """A simple stateless LogSoftmax module."""
    axis: int
    
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis
        
    def forward(self, x: Tensor) -> Tensor:
        return x.log_softmax(axis=self.axis)

# --- Loss Function ---
# nll_loss is imported from autograd at the top of the file