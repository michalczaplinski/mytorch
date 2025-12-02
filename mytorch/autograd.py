from __future__ import annotations

import numpy as np
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, override

# --- Function Base Class ---

class Function:
    """
    Base class for all operations.
    Handles the graph creation and provides forward/backward methods.
    """
    
    input_tensors: tuple[Tensor, ...]
    """The input tensors that were used to create the output tensor
    (e.g., if output = a + b, then inputs = (a, b)). """
    
    saved_tensors: tuple[np.ndarray, ...]
    """The tensors that were saved for backward pass (e.g., x, y for Mul). """
    
    @classmethod
    def apply(cls, *inputs: Tensor, **kwargs: Any) -> Tensor:
        """
        Runs the forward pass and connects the graph.
        'inputs' must be Tensors.
        'kwargs' are non-Tensor arguments (e.g., axis for sum).
        """
        ctx = cls()
        ctx.input_tensors = inputs
        
        input_data = [t.data for t in inputs]
        raw_output = ctx.forward(*input_data, **kwargs)
        # Subclasses should call save_for_backward explicitly if needed

        requires_grad = any(t.requires_grad for t in inputs)
        output_tensor = Tensor(raw_output, requires_grad=requires_grad, _creator=ctx)
        
        return output_tensor

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Computes gradients for inputs and accumulates them.
        """
        input_grads = self.compute_input_grads(grad_output)
        
        if not isinstance(input_grads, tuple):
            input_grads = (input_grads,)
            
        for tensor, grad in zip(self.input_tensors, input_grads):
            if tensor.requires_grad and grad is not None:
                if tensor.grad is None:
                    tensor.grad = np.zeros_like(tensor.data)
                
                tensor.grad += self._unbroadcast_grad(tensor.shape, grad)

    def _unbroadcast_grad(self, target_shape: tuple[int, ...], grad: np.ndarray) -> np.ndarray:
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

    def save_for_backward(self, *args: Any) -> None:
        """(Optional) Save intermediate values needed for backprop."""
        self.saved_tensors = args

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """(Subclass must implement) Performs the actual computation."""
        raise NotImplementedError

    @abstractmethod
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray | tuple[np.ndarray | None, ...]:
        """(Subclass must implement) The raw gradient logic."""
        raise NotImplementedError

class Tensor:
    """
    A simple wrapper for np.ndarray that supports automatic differentiation.
    """
    
    data: np.ndarray
    requires_grad: bool
    grad: np.ndarray | None
    
    _creator: Function | None
    """The Function object (e.g., Add, Mul, etc.) that created 
    this Tensor in the forward pass (or None if leaf tensor). """
    
    def __init__(
        self, 
        data: np.ndarray | Sequence[Any] | float | int, 
        requires_grad: bool = False, 
        _creator: Function | None = None
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        
        self.grad = None
        self._creator = _creator

    def backward(self, grad_output: np.ndarray | float | int | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require grad")

        if self._creator is None:
            return # Root tensor

        # We set the gradient to 1 if it is a scalar and None otherwise
        if grad_output is None:
            if self.data.size != 1:
                raise RuntimeError("grad_output must be specified for non-scalar Tensors")
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(grad_output, dtype=np.float32)

        # Build a topological sort of the graph
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        # Helper function to build the topological sort
        def build_topo(tensor: Tensor) -> None:
            if tensor in visited:
                return

            # Mark as visited to avoid cycles
            visited.add(tensor)

            # Process inputs first (ensures topological order)
            if tensor._creator:
                for inp in tensor._creator.input_tensors:
                    build_topo(inp)
                topo.append(tensor)
        
        build_topo(self)

        # Apply backward pass in reverse topological order
        for tensor in reversed(topo):
            if tensor._creator and tensor.grad is not None:
                tensor._creator.backward(tensor.grad)

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def _as_tensor(self, other: Tensor | np.ndarray | Sequence[Any] | float | int) -> Tensor:
        """Helper to convert NumPy arrays or scalars to Tensors."""
        if not isinstance(other, Tensor):
            return Tensor(other)
        return other

    # --- Overloaded Operators ---
    def __add__(self, other: Tensor | np.ndarray | Sequence[Any] | float | int) -> Tensor:
        return Add.apply(self, self._as_tensor(other))
    
    def __mul__(self, other: Tensor | np.ndarray | Sequence[Any] | float | int) -> Tensor:
        return Mul.apply(self, self._as_tensor(other))
    
    def __matmul__(self, other: Tensor | np.ndarray | Sequence[Any] | float | int) -> Tensor:
        return MatMul.apply(self, self._as_tensor(other))
    
    def __neg__(self) -> Tensor:
        return Neg.apply(self)
    
    def __sub__(self, other: Tensor | np.ndarray | Sequence[Any] | float | int) -> Tensor:
        return self + (-self._as_tensor(other))
    
    def __pow__(self, other: float | int) -> Tensor:
        return Pow.apply(self, c=other) # Pass scalar as kwarg
    
    def __truediv__(self, other: Tensor | np.ndarray | Sequence[Any] | float | int) -> Tensor:
        return Div.apply(self, self._as_tensor(other))

    # --- Standard Methods ---
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        return Sum.apply(self, axis=axis, keepdims=keepdims)
    
    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        return Mean.apply(self, axis=axis, keepdims=keepdims)
    
    def var(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        return Var.apply(self, axis=axis, keepdims=keepdims)
    
    def relu(self) -> Tensor:
        return ReLU.apply(self)
    
    def log_softmax(self, axis: int = -1) -> Tensor:
        return LogSoftmax.apply(self, axis=axis)

    def __getitem__(self, indices: np.ndarray | Sequence[int] | int) -> Tensor:
        return Indexing.apply(self, indices=indices)

    def reshape(self, *shape: int) -> Tensor:
        return Reshape.apply(self, new_shape=shape)
    
    def transpose(self, *axes: int) -> Tensor:
        return Transpose.apply(self, axes=axes)

    # --- Properties ---
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

class Add(Function):
    @override
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output

class Mul(Function):
    @override
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return x * y
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.saved_tensors
        return grad_output * y, grad_output * x

class Div(Function):
    @override
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return x / y
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.saved_tensors
        grad_x = grad_output / y           # ∂(x/y)/∂x = 1/y
        grad_y = -grad_output * x / (y * y)  # ∂(x/y)/∂y = -x/y²
        return grad_x, grad_y

class Neg(Function):
    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        return -grad_output

class Pow(Function):
    c: float | int
    
    @override
    def forward(self, x: np.ndarray, c: float | int) -> np.ndarray:
        self.save_for_backward(x)
        self.c = c
        return x ** c
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        x, = self.saved_tensors
        return grad_output * (self.c * (x ** (self.c - 1)))

class MatMul(Function):
    @override
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return x @ y
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.saved_tensors
        grad_x = grad_output @ y.T
        grad_y = x.T @ grad_output
        return grad_x, grad_y

class ReLU(Function):
    mask: np.ndarray
    
    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0)
        return x * self.mask
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask

class Sum(Function):
    input_shape: tuple[int, ...]
    axis: int | tuple[int, ...] | None
    keepdims: bool
    
    @override
    def forward(
        self, 
        x: np.ndarray, 
        axis: int | tuple[int, ...] | None = None, 
        keepdims: bool = False
    ) -> np.ndarray:
        self.input_shape = x.shape
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        return np.broadcast_to(grad_output, self.input_shape)

class Mean(Function):
    input_shape: tuple[int, ...]
    n: float
    axis: int | tuple[int, ...] | None
    keepdims: bool
    
    @override
    def forward(
        self, 
        x: np.ndarray, 
        axis: int | tuple[int, ...] | None = None, 
        keepdims: bool = False
    ) -> np.ndarray:
        self.input_shape = x.shape
        output_shape = np.mean(x, axis=axis, keepdims=keepdims).shape
        self.n = np.prod(self.input_shape) / np.prod(output_shape)
        self.axis = axis
        self.keepdims = keepdims
        return np.mean(x, axis=axis, keepdims=keepdims)
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        return np.broadcast_to(grad_output, self.input_shape) / self.n

class Var(Function):
    input_shape: tuple[int, ...]
    axis: int | tuple[int, ...] | None
    keepdims: bool
    
    @override
    def forward(self, x: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray:
        self.input_shape = x.shape
        self.axis = axis
        self.keepdims = keepdims
        return np.var(x, axis=axis, keepdims=keepdims)
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        return np.broadcast_to(grad_output, self.input_shape)

class LogSoftmax(Function):
    axis: int
    
    @override
    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        self.axis = axis
        max_x = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - max_x)
        sum_exp_x = exp_x.sum(axis=axis, keepdims=True)
        log_probs = (x - max_x) - np.log(sum_exp_x)
        self.save_for_backward(log_probs)
        return log_probs
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        log_probs, = self.saved_tensors
        softmax_output = np.exp(log_probs)
        grad_sum = np.sum(grad_output, axis=self.axis, keepdims=True)
        return grad_output - (softmax_output * grad_sum)

class Indexing(Function):
    """Indexing operation for embedding lookups (weight[indices])."""
    input_shape: tuple[int, ...]
    indices: np.ndarray[Any, np.dtype[np.intp]]
    
    @override
    def forward(self, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        self.indices = np.asarray(indices, dtype=np.intp)  # Ensure integer type for indexing
        return x[self.indices]
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        grad_x = np.zeros(self.input_shape, dtype=np.float32)
        np.add.at(grad_x, self.indices, grad_output)
        return grad_x

class NLLLoss(Function):
    n_samples: int
    n_classes: int
    targets: np.ndarray
    
    @override
    def forward(self, log_probs: np.ndarray, **kwargs: Any) -> np.ndarray:
        targets = kwargs.get('targets')
        if targets is None:
            raise ValueError("targets must be provided")
        self.targets = targets
        n_samples, n_classes = log_probs.shape
        self.n_samples, self.n_classes = n_samples, n_classes
        correct_log_probs = log_probs[range(n_samples), targets]
        return -correct_log_probs.mean()
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> tuple[np.ndarray, None]:
        grad_log_probs = np.zeros((self.n_samples, self.n_classes), dtype=np.float32)
        grad_log_probs[range(self.n_samples), self.targets] = -1.0 / self.n_samples
        return grad_log_probs * grad_output, None # No grad for targets

class Reshape(Function):
    input_shape: tuple[int, ...]
    new_shape: tuple[int, ...]
    
    @override
    def forward(self, x: np.ndarray, new_shape: tuple[int, ...]) -> np.ndarray:
        self.input_shape = x.shape
        self.new_shape = new_shape
        return x.reshape(new_shape)
    
    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self.input_shape)

class Transpose(Function):
    axes: tuple[int, ...]

    @override
    def forward(self, x: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        self.axes = axes
        return x.transpose(axes)

    @override
    def compute_input_grads(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.transpose(self.axes)