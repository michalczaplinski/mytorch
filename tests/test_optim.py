"""
Tests for mytorch.optim module
"""
import numpy as np
import pytest
from mytorch.autograd import Tensor
from mytorch.nn import Linear
from mytorch.optim import SGD


class TestSGD:
    """Test SGD optimizer"""
    
    def test_sgd_initialization(self):
        """Test SGD initialization"""
        params = [Tensor([1, 2], requires_grad=True)]
        optimizer = SGD(params, lr=0.01)
        
        assert optimizer.lr == 0.01
        assert len(optimizer.params) == 1
        
    def test_sgd_step(self):
        """Test SGD parameter update"""
        # Create a simple parameter
        param = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1)
        
        # Set gradient manually
        param.grad = np.array([0.5, 1.0])
        
        # Store original values
        original = param.data.copy()
        
        # Take optimization step
        optimizer.step()
        
        # Parameters should be updated: new = old - lr * grad
        expected = original - 0.1 * np.array([0.5, 1.0])
        assert np.allclose(param.data, expected)
        
    def test_sgd_zero_grad(self):
        """Test zeroing gradients through optimizer"""
        params = [
            Tensor([1, 2], requires_grad=True),
            Tensor([3, 4], requires_grad=True)
        ]
        optimizer = SGD(params, lr=0.01)
        
        # Set gradients
        for p in params:
            p.grad = np.ones_like(p.data)
        
        # Zero them
        optimizer.zero_grad()
        
        # Check all are zero
        for p in params:
            assert np.allclose(p.grad, 0.0)
            
    def test_sgd_with_model(self):
        """Test SGD with a simple model"""
        # Create a linear layer
        layer = Linear(2, 1)
        optimizer = SGD(layer.parameters(), lr=0.1)
        
        # Forward pass
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = layer(x)
        loss = y.sum()
        
        # Store original weights
        original_weight = layer.weight.data.copy()
        original_bias = layer.bias.data.copy()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        # Weights should have changed
        assert not np.allclose(layer.weight.data, original_weight)
        assert not np.allclose(layer.bias.data, original_bias)
        
    def test_sgd_multiple_steps(self):
        """Test multiple optimization steps"""
        param = Tensor([10.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1)
        
        # Simulate gradient descent on f(x) = x^2
        for _ in range(10):
            # Compute gradient: df/dx = 2x
            param.grad = 2 * param.data
            optimizer.step()
        
        # After multiple steps, parameter should decrease
        assert param.data[0] < 10.0
        
    def test_sgd_different_learning_rates(self):
        """Test that different learning rates produce different updates"""
        param1 = Tensor([1.0], requires_grad=True)
        param2 = Tensor([1.0], requires_grad=True)
        
        opt1 = SGD([param1], lr=0.01)
        opt2 = SGD([param2], lr=0.1)
        
        # Same gradient
        param1.grad = np.array([1.0])
        param2.grad = np.array([1.0])
        
        opt1.step()
        opt2.step()
        
        # param2 should have changed more
        assert abs(param2.data[0] - 1.0) > abs(param1.data[0] - 1.0)


class TestOptimizationScenarios:
    """Test realistic optimization scenarios"""
    
    def test_simple_linear_regression(self):
        """Test optimizing a simple linear function"""
        # Target: y = 2x + 1
        layer = Linear(1, 1)
        optimizer = SGD(layer.parameters(), lr=0.01)
        
        # Training data
        X = np.array([[1.0], [2.0], [3.0]])
        y_true = np.array([[3.0], [5.0], [7.0]])
        
        initial_loss = None
        final_loss = None
        
        # Train for a few iterations
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Forward
            y_pred = layer(Tensor(X))
            
            # Simple MSE loss
            diff = y_pred.data - y_true
            loss = Tensor(np.mean(diff ** 2), requires_grad=True)
            
            if epoch == 0:
                initial_loss = loss.data
            if epoch == 99:
                final_loss = loss.data
            
            # Backward (manually compute gradient for this test)
            grad = 2 * diff / len(X)
            y_pred.backward(grad)
            
            optimizer.step()
        
        # Loss should decrease
        assert final_loss < initial_loss
