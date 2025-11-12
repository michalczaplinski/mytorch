from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union, Any, Sequence, Callable

def _unbroadcast_grad(target_shape: Tuple[int, ...], grad: np.ndarray) -> np.ndarray:
    """
    Helper to correctly sum gradients for broadcasted operations.
    """
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    if grad.shape != target_shape:
        grad = grad.reshape(target_shape)
    return grad

class Tensor:
    """
    A simple wrapper for np.ndarray that supports automatic differentiation.
    """
    
    data: np.ndarray
    requires_grad: bool
    grad: Optional[np.ndarray]
    _parents: Tuple[Tensor, ...]
    _backward: Optional[Callable[[np.ndarray], None]]
    
    def __init__(
        self, 
        data: Union[np.ndarray, Sequence[Any], float, int], 
        requires_grad: bool = False,
        _parents: Tuple[Tensor, ...] = (),
        _backward: Optional[Callable[[np.ndarray], None]] = None
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        
        self.grad = None
        self._parents = _parents
        self._backward = _backward

    def backward(self, grad_output: Optional[Union[np.ndarray, float, int]] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require grad")

        if self._backward is None:
            return # Root tensor

        if grad_output is None:
            if self.data.size != 1:
                raise RuntimeError("grad_output must be specified for non-scalar Tensors")
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(grad_output, dtype=np.float32)

        # Build a topological sort of the graph
        topo: list[Tensor] = []
        visited: set[Tensor] = set()
        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                if v._parents:
                    for parent in v._parents:
                        build_topo(parent)
                    topo.append(v)
        
        build_topo(self)

        # Apply backward pass in reverse topological order
        for v in reversed(topo):
            if v._backward and v.grad is not None:
                v._backward(v.grad)

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def _as_tensor(self, other: Union[Tensor, np.ndarray, Sequence[Any], float, int]) -> Tensor:
        """Helper to convert NumPy arrays or scalars to Tensors."""
        if not isinstance(other, Tensor):
            return Tensor(other)
        return other

    # --- Overloaded Operators ---
    def __add__(self, other: Union[Tensor, np.ndarray, Sequence[Any], float, int]) -> Tensor:
        return _add(self, self._as_tensor(other))
    
    def __mul__(self, other: Union[Tensor, np.ndarray, Sequence[Any], float, int]) -> Tensor:
        return _mul(self, self._as_tensor(other))
    
    def __matmul__(self, other: Union[Tensor, np.ndarray, Sequence[Any], float, int]) -> Tensor:
        return _matmul(self, self._as_tensor(other))
    
    def __neg__(self) -> Tensor:
        return _neg(self)
    
    def __sub__(self, other: Union[Tensor, np.ndarray, Sequence[Any], float, int]) -> Tensor:
        return self + (-other)
    
    def __pow__(self, other: Union[float, int]) -> Tensor:
        return _pow(self, other) # Assumes 'other' is scalar

    # --- Standard Methods ---
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        return _sum(self, axis=axis, keepdims=keepdims)
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        return _mean(self, axis=axis, keepdims=keepdims)
    
    def relu(self) -> Tensor:
        return _relu(self)
    
    def log_softmax(self, axis: int = -1) -> Tensor:
        return _log_softmax(self, axis=axis)

    # --- Properties ---
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

# --- Operation Implementations ---

def _add(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise addition."""
    requires_grad = x.requires_grad or y.requires_grad
    out_data = x.data + y.data
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += _unbroadcast_grad(x.shape, grad_output)
        if y.requires_grad:
            if y.grad is None:
                y.grad = np.zeros_like(y.data)
            y.grad += _unbroadcast_grad(y.shape, grad_output)
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x, y), _backward=backward if requires_grad else None)

def _mul(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise multiplication."""
    requires_grad = x.requires_grad or y.requires_grad
    out_data = x.data * y.data
    
    # Save values needed for backward
    x_data = x.data
    y_data = y.data
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += _unbroadcast_grad(x.shape, grad_output * y_data)
        if y.requires_grad:
            if y.grad is None:
                y.grad = np.zeros_like(y.data)
            y.grad += _unbroadcast_grad(y.shape, grad_output * x_data)
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x, y), _backward=backward if requires_grad else None)

def _neg(x: Tensor) -> Tensor:
    """Negation."""
    requires_grad = x.requires_grad
    out_data = -x.data
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += -grad_output
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x,), _backward=backward if requires_grad else None)

def _pow(x: Tensor, c: Union[float, int]) -> Tensor:
    """Power operation (x^c)."""
    requires_grad = x.requires_grad
    out_data = x.data ** c
    
    # Save values needed for backward
    x_data = x.data
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += grad_output * (c * (x_data ** (c - 1)))
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x,), _backward=backward if requires_grad else None)

def _matmul(x: Tensor, y: Tensor) -> Tensor:
    """Matrix multiplication."""
    requires_grad = x.requires_grad or y.requires_grad
    out_data = x.data @ y.data
    
    # Save values needed for backward
    x_data = x.data
    y_data = y.data
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += grad_output @ y_data.T
        if y.requires_grad:
            if y.grad is None:
                y.grad = np.zeros_like(y.data)
            y.grad += x_data.T @ grad_output
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x, y), _backward=backward if requires_grad else None)

def _relu(x: Tensor) -> Tensor:
    """ReLU activation."""
    requires_grad = x.requires_grad
    mask = (x.data > 0)
    out_data = x.data * mask
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += grad_output * mask
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x,), _backward=backward if requires_grad else None)

def _sum(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Sum reduction."""
    requires_grad = x.requires_grad
    input_shape = x.shape
    out_data = np.sum(x.data, axis=axis, keepdims=keepdims)
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            # Expand dimensions if needed
            if not keepdims and axis is not None:
                grad_output = np.expand_dims(grad_output, axis)
            x.grad += np.broadcast_to(grad_output, input_shape)
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x,), _backward=backward if requires_grad else None)

def _mean(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Mean reduction."""
    requires_grad = x.requires_grad
    input_shape = x.shape
    output_shape = np.mean(x.data, axis=axis, keepdims=keepdims).shape
    n = np.prod(input_shape) / np.prod(output_shape)
    out_data = np.mean(x.data, axis=axis, keepdims=keepdims)
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            # Expand dimensions if needed
            if not keepdims and axis is not None:
                grad_output = np.expand_dims(grad_output, axis)
            x.grad += np.broadcast_to(grad_output, input_shape) / n
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x,), _backward=backward if requires_grad else None)

def _log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Log softmax activation."""
    requires_grad = x.requires_grad
    max_x = x.data.max(axis=axis, keepdims=True)
    exp_x = np.exp(x.data - max_x)
    sum_exp_x = exp_x.sum(axis=axis, keepdims=True)
    log_probs = (x.data - max_x) - np.log(sum_exp_x)
    out_data = log_probs
    
    def backward(grad_output: np.ndarray) -> None:
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            softmax_output = np.exp(log_probs)
            grad_sum = np.sum(grad_output, axis=axis, keepdims=True)
            x.grad += grad_output - (softmax_output * grad_sum)
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(x,), _backward=backward if requires_grad else None)

def nll_loss(log_probs: Tensor, targets: np.ndarray) -> Tensor:
    """
    Computes the Negative Log Likelihood loss.
    """
    N, C = log_probs.shape
    correct_log_probs = log_probs.data[range(N), targets]
    out_data = -correct_log_probs.mean()
    requires_grad = log_probs.requires_grad
    
    def backward(grad_output: np.ndarray) -> None:
        if log_probs.requires_grad:
            if log_probs.grad is None:
                log_probs.grad = np.zeros_like(log_probs.data)
            grad_log_probs = np.zeros((N, C), dtype=np.float32)
            grad_log_probs[range(N), targets] = -1.0 / N
            log_probs.grad += grad_log_probs * grad_output
    
    return Tensor(out_data, requires_grad=requires_grad, _parents=(log_probs,), _backward=backward if requires_grad else None)
