# MyTorch

A lightweight deep learning framework built from scratch using NumPy, featuring automatic differentiation and neural network modules.

## Features

- **Automatic Differentiation**: Reverse-mode automatic differentiation (backpropagation) for computing gradients
- **Neural Network Modules**: Pre-built layers (Linear, ReLU, LogSoftmax) and loss functions
- **Optimizers**: SGD optimizer with extensible architecture
- **Pure NumPy**: No external ML frameworks required, only NumPy

## Installation

```bash
pip install numpy>=2.3.4
```

## Quick Start

### Basic Tensor Operations

```python
from mytorch.autograd import Tensor

# Create tensors with gradient tracking
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], requires_grad=True)

# Perform operations
z = x + y
w = x * y
loss = (z * w).sum()

# Compute gradients
loss.backward()
print(x.grad)  # Gradients computed automatically
```

### Building a Neural Network

```python
from mytorch.nn import Module, Linear, ReLU, LogSoftmax
from mytorch.optim import SGD
from mytorch.autograd import Tensor, nll_loss
import numpy as np

# Define a simple MLP
class SimpleMLP(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)
        self.log_softmax = LogSoftmax(axis=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

# Create model and optimizer
model = SimpleMLP(784, 128, 10)
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    # Forward pass
    predictions = model(Tensor(X_train))
    loss = nll_loss(predictions, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.data.item()}")
```

## Supported Operations

### Tensor Operations
- Addition: `x + y`
- Multiplication: `x * y`
- Matrix multiplication: `x @ y`
- Negation: `-x`
- Subtraction: `x - y`
- Power: `x ** 2`
- Sum: `x.sum(axis=0)`
- Mean: `x.mean(axis=1)`

### Activations
- ReLU: `x.relu()`
- LogSoftmax: `x.log_softmax(axis=-1)`

### Loss Functions
- Negative Log Likelihood: `nll_loss(predictions, targets)`

## Architecture

- `mytorch.autograd`: Core automatic differentiation engine with `Tensor` class
- `mytorch.nn`: Neural network building blocks (`Module`, `Linear`, activation functions)
- `mytorch.optim`: Optimization algorithms (SGD)

## Example: MNIST Training

See `examples/mnist/mnist.py` for a complete MNIST training example.

## Project Structure

```
mytorch/
├── mytorch/
│   ├── __init__.py
│   ├── autograd.py    # Automatic differentiation engine
│   ├── nn.py          # Neural network modules
│   └── optim.py       # Optimizers
├── examples/
│   └── mnist/         # MNIST training example
├── README.md
└── pyproject.toml
```

## Development

This is an educational project designed to demonstrate the fundamentals of automatic differentiation and neural networks. It is not intended for production use.

## License

This project is open source and available for educational purposes.
