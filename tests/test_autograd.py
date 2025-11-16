"""
Tests for mytorch.autograd module
"""
import numpy as np
import pytest
from mytorch.autograd import Tensor, nll_loss


class TestTensorBasics:
    """Test basic Tensor functionality"""
    
    def test_tensor_creation(self):
        """Test creating tensors from various inputs"""
        # From list
        t1 = Tensor([1, 2, 3])
        assert t1.data.shape == (3,)
        assert t1.dtype == np.float32
        
        # From numpy array
        t2 = Tensor(np.array([1.0, 2.0]))
        assert t2.data.shape == (2,)
        
        # From scalar
        t3 = Tensor(5.0)
        assert t3.data.shape == ()
        
    def test_requires_grad(self):
        """Test gradient tracking flag"""
        t1 = Tensor([1, 2], requires_grad=True)
        assert t1.requires_grad is True
        
        t2 = Tensor([1, 2], requires_grad=False)
        assert t2.requires_grad is False
        
    def test_zero_grad(self):
        """Test zeroing gradients"""
        t = Tensor([1, 2], requires_grad=True)
        t.grad = np.array([3.0, 4.0])
        t.zero_grad()
        assert np.allclose(t.grad, [0.0, 0.0])


class TestTensorOperations:
    """Test tensor arithmetic operations"""
    
    def test_addition(self):
        """Test element-wise addition"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([4, 5, 6], requires_grad=True)
        z = x + y
        
        assert np.allclose(z.data, [5, 7, 9])
        assert z.requires_grad is True
        
    def test_multiplication(self):
        """Test element-wise multiplication"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([2, 3, 4], requires_grad=True)
        z = x * y
        
        assert np.allclose(z.data, [2, 6, 12])
        assert z.requires_grad is True
        
    def test_negation(self):
        """Test negation"""
        x = Tensor([1, -2, 3], requires_grad=True)
        y = -x
        
        assert np.allclose(y.data, [-1, 2, -3])
        
    def test_subtraction(self):
        """Test subtraction"""
        x = Tensor([5, 6, 7], requires_grad=True)
        y = Tensor([1, 2, 3], requires_grad=True)
        z = x - y
        
        assert np.allclose(z.data, [4, 4, 4])
        
    def test_power(self):
        """Test power operation"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = x ** 2
        
        assert np.allclose(y.data, [1, 4, 9])
        
    def test_division(self):
        """Test division"""
        x = Tensor([4, 6, 8], requires_grad=True)
        y = Tensor([2, 3, 4], requires_grad=True)
        z = x / y
        
        assert np.allclose(z.data, [2, 2, 2])
        assert z.requires_grad is True
        
    def test_matmul(self):
        """Test matrix multiplication"""
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = Tensor([[5, 6], [7, 8]], requires_grad=True)
        z = x @ y
        
        expected = np.array([[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]])
        assert np.allclose(z.data, expected)
        
    def test_sum(self):
        """Test sum reduction"""
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        
        # Sum all
        y = x.sum()
        assert np.allclose(y.data, 10)
        
        # Sum along axis
        z = x.sum(axis=0)
        assert np.allclose(z.data, [4, 6])
        
    def test_mean(self):
        """Test mean reduction"""
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        
        # Mean all
        y = x.mean()
        assert np.allclose(y.data, 2.5)
        
        # Mean along axis
        z = x.mean(axis=1)
        assert np.allclose(z.data, [1.5, 3.5])
        
    def test_relu(self):
        """Test ReLU activation"""
        x = Tensor([-1, 0, 1, 2], requires_grad=True)
        y = x.relu()
        
        assert np.allclose(y.data, [0, 0, 1, 2])
        
    def test_sigmoid(self):
        """Test Sigmoid activation"""
        x = Tensor([0, 1, -1], requires_grad=True)
        y = x.sigmoid()
        
        # Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.73, Sigmoid(-1) ≈ 0.27
        assert np.allclose(y.data[0], 0.5)
        assert y.data[1] > 0.7 and y.data[1] < 0.8
        assert y.data[2] > 0.2 and y.data[2] < 0.3
        
    def test_tanh(self):
        """Test Tanh activation"""
        x = Tensor([0, 1, -1], requires_grad=True)
        y = x.tanh()
        
        # Tanh(0) = 0, Tanh(1) ≈ 0.76, Tanh(-1) ≈ -0.76
        assert np.allclose(y.data[0], 0.0)
        assert y.data[1] > 0.7 and y.data[1] < 0.8
        assert y.data[2] > -0.8 and y.data[2] < -0.7
        
    def test_reverse_add(self):
        """Test reverse addition (scalar + tensor)"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = 5 + x
        
        assert np.allclose(y.data, [6, 7, 8])
        
    def test_reverse_mul(self):
        """Test reverse multiplication (scalar * tensor)"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = 2 * x
        
        assert np.allclose(y.data, [2, 4, 6])
        
    def test_reverse_sub(self):
        """Test reverse subtraction (scalar - tensor)"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = 10 - x
        
        assert np.allclose(y.data, [9, 8, 7])
        
    def test_reverse_div(self):
        """Test reverse division (scalar / tensor)"""
        x = Tensor([2, 4, 5], requires_grad=True)
        y = 10 / x
        
        assert np.allclose(y.data, [5, 2.5, 2])
        
    def test_log_softmax(self):
        """Test log softmax"""
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = x.log_softmax(axis=1)
        
        # Should sum to approximately 1 when exponentiated
        assert np.allclose(np.exp(y.data).sum(), 1.0)


class TestBackpropagation:
    """Test automatic differentiation"""
    
    def test_simple_backward(self):
        """Test backward on simple scalar operation"""
        x = Tensor([2.0], requires_grad=True)
        y = x * x
        y.backward()
        
        # dy/dx = 2x = 2*2 = 4
        assert np.allclose(x.grad, [4.0])
        
    def test_addition_backward(self):
        """Test gradients for addition"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([4, 5, 6], requires_grad=True)
        z = x + y
        z.backward(np.ones(3))
        
        assert np.allclose(x.grad, [1, 1, 1])
        assert np.allclose(y.grad, [1, 1, 1])
        
    def test_multiplication_backward(self):
        """Test gradients for multiplication"""
        x = Tensor([2, 3], requires_grad=True)
        y = Tensor([4, 5], requires_grad=True)
        z = x * y
        z.backward(np.ones(2))
        
        # dz/dx = y, dz/dy = x
        assert np.allclose(x.grad, [4, 5])
        assert np.allclose(y.grad, [2, 3])
        
    def test_power_backward(self):
        """Test gradients for power operation"""
        x = Tensor([2, 3], requires_grad=True)
        y = x ** 2
        y.backward(np.ones(2))
        
        # dy/dx = 2x
        assert np.allclose(x.grad, [4, 6])
        
    def test_division_backward(self):
        """Test gradients for division"""
        x = Tensor([4, 6], requires_grad=True)
        y = Tensor([2, 3], requires_grad=True)
        z = x / y
        z.backward(np.ones(2))
        
        # dz/dx = 1/y, dz/dy = -x/y^2
        assert np.allclose(x.grad, [0.5, 1.0/3.0])
        assert np.allclose(y.grad, [-1.0, -2.0/3.0])
        
    def test_matmul_backward(self):
        """Test gradients for matrix multiplication"""
        x = Tensor([[1, 2]], requires_grad=True)
        y = Tensor([[3], [4]], requires_grad=True)
        z = x @ y
        z.backward()
        
        assert x.grad.shape == (1, 2)
        assert y.grad.shape == (2, 1)
        
    def test_relu_backward(self):
        """Test gradients for ReLU"""
        x = Tensor([-1, 0, 1], requires_grad=True)
        y = x.relu()
        y.backward(np.ones(3))
        
        # Gradient is 0 for negative inputs, 1 for positive
        assert np.allclose(x.grad, [0, 0, 1])
        
    def test_sigmoid_backward(self):
        """Test gradients for Sigmoid"""
        x = Tensor([0.0], requires_grad=True)
        y = x.sigmoid()
        y.backward()
        
        # Gradient at x=0: σ(0) * (1 - σ(0)) = 0.5 * 0.5 = 0.25
        assert np.allclose(x.grad, [0.25])
        
    def test_tanh_backward(self):
        """Test gradients for Tanh"""
        x = Tensor([0.0], requires_grad=True)
        y = x.tanh()
        y.backward()
        
        # Gradient at x=0: 1 - tanh(0)^2 = 1 - 0 = 1
        assert np.allclose(x.grad, [1.0])
        
    def test_chain_rule(self):
        """Test chain rule through multiple operations"""
        x = Tensor([2.0], requires_grad=True)
        y = x * x  # x^2
        z = y + y  # 2x^2
        w = z * z  # 4x^4
        w.backward()
        
        # dw/dx = 16x^3 = 16 * 8 = 128
        assert np.allclose(x.grad, [128.0], rtol=1e-5)
        
    def test_no_grad_error(self):
        """Test error when calling backward on non-grad tensor"""
        x = Tensor([1, 2], requires_grad=False)
        with pytest.raises(RuntimeError):
            x.backward()
            
    def test_broadcast_add(self):
        """Test broadcasting in addition"""
        x = Tensor([[1, 2]], requires_grad=True)  # Shape (1, 2)
        y = Tensor([3], requires_grad=True)  # Shape (1,)
        z = x + y
        z.backward(np.ones((1, 2)))
        
        assert x.grad.shape == (1, 2)
        assert y.grad.shape == (1,)


class TestLossFunctions:
    """Test loss functions"""
    
    def test_nll_loss(self):
        """Test negative log likelihood loss"""
        # Batch of 3 samples, 4 classes
        log_probs = Tensor([
            [-1.0, -2.0, -3.0, -4.0],
            [-2.0, -1.0, -3.0, -4.0],
            [-3.0, -2.0, -1.0, -4.0]
        ], requires_grad=True)
        
        targets = np.array([0, 1, 2])  # Target classes
        
        loss = nll_loss(log_probs, targets)
        
        # Loss should be mean of -log_probs at target indices
        expected = -(-1.0 - 1.0 - 1.0) / 3
        assert np.allclose(loss.data, expected)
        
        # Test backward
        loss.backward()
        assert log_probs.grad is not None
        assert log_probs.grad.shape == (3, 4)
