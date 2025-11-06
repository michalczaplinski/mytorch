# autograd.py

import numpy as np

# --- Function Base Class ---

class Function:
    """
    Base class for all operations.
    Handles the graph creation and provides forward/backward methods.
    """
    
    @classmethod
    def apply(cls, *inputs, **kwargs):
        """
        Runs the forward pass and connects the graph.
        'inputs' must be Tensors.
        'kwargs' are non-Tensor arguments (e.g., axis for sum).
        """
        ctx = cls()
        ctx.inputs = inputs
        
        input_data = [t.data for t in inputs]
        raw_output = ctx.forward(*input_data, **kwargs)
        # Subclasses should call save_for_backward explicitly if needed

        requires_grad = any(t.requires_grad for t in inputs)
        output_tensor = Tensor(raw_output, requires_grad=requires_grad, _creator=ctx)
        
        return output_tensor

    def backward(self, grad_output):
        """
        Computes gradients for inputs and accumulates them.
        """
        input_grads = self.compute_input_grads(grad_output)
        
        if not isinstance(input_grads, tuple):
            input_grads = (input_grads,)
            
        for tensor, grad in zip(self.inputs, input_grads):
            if tensor.requires_grad and grad is not None:
                if tensor.grad is None:
                    tensor.grad = np.zeros_like(tensor.data)
                
                tensor.grad += self._unbroadcast_grad(tensor.shape, grad)

    def _unbroadcast_grad(self, target_shape, grad):
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

    def save_for_backward(self, *args):
        """(Optional) Save intermediate values needed for backprop."""
        self.saved_tensors = args

    def forward(self, *args, **kwargs):
        """(Subclass must implement) Performs the actual computation."""
        raise NotImplementedError

    def compute_input_grads(self, grad_output):
        """(Subclass must implement) The raw gradient logic."""
        raise NotImplementedError

# --- Tensor Class ---

class Tensor:
    """
    A simple wrapper for np.ndarray that supports automatic differentiation.
    """
    
    def __init__(self, data, requires_grad=False, _creator=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        
        self.grad = None
        self.grad_fn = _creator
        self._ctx = _creator # The Function object that created this

    def backward(self, grad_output=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require grad")

        if self.grad_fn is None:
            return # Root tensor

        if grad_output is None:
            if self.data.size != 1:
                raise RuntimeError("grad_output must be specified for non-scalar Tensors")
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(grad_output, dtype=np.float32)

        # Build a topological sort of the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._ctx:
                    for inp in v._ctx.inputs:
                        build_topo(inp)
                    topo.append(v)
        
        build_topo(self)

        # Apply backward pass in reverse topological order
        for v in reversed(topo):
            if v._ctx:
                v._ctx.backward(v.grad)

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0.0)

    def _as_tensor(self, other):
        """Helper to convert NumPy arrays or scalars to Tensors."""
        if not isinstance(other, Tensor):
            return Tensor(other)
        return other

    # --- Overloaded Operators ---
    def __add__(self, other): return Add.apply(self, self._as_tensor(other))
    def __mul__(self, other): return Mul.apply(self, self._as_tensor(other))
    def __matmul__(self, other): return MatMul.apply(self, self._as_tensor(other))
    def __neg__(self): return Neg.apply(self)
    def __sub__(self, other): return self + (-other)
    def __pow__(self, other): return Pow.apply(self, other) # Assumes 'other' is scalar

    # --- Standard Methods ---
    def sum(self, axis=None, keepdims=False): return Sum.apply(self, axis=axis, keepdims=keepdims)
    def mean(self, axis=None, keepdims=False): return Mean.apply(self, axis=axis, keepdims=keepdims)
    def relu(self): return ReLU.apply(self)
    def log_softmax(self, axis=-1): return LogSoftmax.apply(self, axis=axis)

    # --- Properties ---
    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    def __repr__(self): return f"Tensor({self.data}, requires_grad={self.requires_grad})"

# --- Function Subclasses ---

class Add(Function):
    def forward(self, x, y): return x + y
    def compute_input_grads(self, grad_output): return grad_output, grad_output

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x * y
    def compute_input_grads(self, grad_output):
        x, y = self.saved_tensors
        return grad_output * y, grad_output * x

class Neg(Function):
    def forward(self, x): return -x
    def compute_input_grads(self, grad_output): return -grad_output

class Pow(Function):
    def forward(self, x, c):
        self.save_for_backward(x)
        self.c = c
        return x ** c
    def compute_input_grads(self, grad_output):
        x, = self.saved_tensors
        return grad_output * (self.c * (x ** (self.c - 1)))

class MatMul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x @ y
    def compute_input_grads(self, grad_output):
        x, y = self.saved_tensors
        grad_x = grad_output @ y.T
        grad_y = x.T @ grad_output
        return grad_x, grad_y

class ReLU(Function):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def compute_input_grads(self, grad_output):
        return grad_output * self.mask

class Sum(Function):
    def forward(self, x, axis=None, keepdims=False):
        self.input_shape = x.shape
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(x, axis=axis, keepdims=keepdims)
    def compute_input_grads(self, grad_output):
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        return np.broadcast_to(grad_output, self.input_shape)

class Mean(Function):
     def forward(self, x, axis=None, keepdims=False):
        self.input_shape = x.shape
        output_shape = np.mean(x, axis=axis, keepdims=keepdims).shape
        self.n = np.prod(self.input_shape) / np.prod(output_shape)
        self.axis = axis
        self.keepdims = keepdims
        return np.mean(x, axis=axis, keepdims=keepdims)
     def compute_input_grads(self, grad_output):
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        return np.broadcast_to(grad_output, self.input_shape) / self.n

class LogSoftmax(Function):
    def forward(self, x, axis=-1):
        self.axis = axis
        max_x = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - max_x)
        sum_exp_x = exp_x.sum(axis=axis, keepdims=True)
        log_probs = (x - max_x) - np.log(sum_exp_x)
        self.save_for_backward(log_probs)
        return log_probs
    def compute_input_grads(self, grad_output):
        log_probs, = self.saved_tensors
        softmax_output = np.exp(log_probs)
        grad_sum = np.sum(grad_output, axis=self.axis, keepdims=True)
        return grad_output - (softmax_output * grad_sum)

class NLLLoss(Function):
    def forward(self, log_probs, targets):
        N, C = log_probs.shape
        self.N, self.C = N, C
        self.targets = targets
        correct_log_probs = log_probs[range(N), targets]
        return -correct_log_probs.mean()
    def compute_input_grads(self, grad_output):
        grad_log_probs = np.zeros((self.N, self.C), dtype=np.float32)
        grad_log_probs[range(self.N), self.targets] = -1.0 / self.N
        return grad_log_probs * grad_output, None # No grad for targets

# --- nn.Module Base Class ---

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