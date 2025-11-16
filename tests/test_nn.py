"""
Tests for mytorch.nn module
"""
import numpy as np
import pytest
from mytorch.autograd import Tensor
from mytorch.nn import Module, Linear, ReLU, LogSoftmax


class TestModule:
    """Test Module base class"""
    
    def test_module_call(self):
        """Test that modules are callable"""
        class SimpleModule(Module):
            def forward(self, x):
                return x * 2
        
        m = SimpleModule()
        x = Tensor([1, 2, 3])
        y = m(x)
        
        assert np.allclose(y.data, [2, 4, 6])
        
    def test_parameters(self):
        """Test parameter collection"""
        linear = Linear(3, 2)
        params = list(linear.parameters())
        
        # Should have 2 parameters: weight and bias
        assert len(params) == 2
        assert all(isinstance(p, Tensor) for p in params)
        assert all(p.requires_grad for p in params)
        
    def test_zero_grad(self):
        """Test zero_grad on module"""
        linear = Linear(2, 1)
        
        # Set gradients manually
        for p in linear.parameters():
            p.grad = np.ones_like(p.data)
        
        # Zero them
        linear.zero_grad()
        
        # Check all are zero
        for p in linear.parameters():
            assert np.allclose(p.grad, 0.0)


class TestLinear:
    """Test Linear layer"""
    
    def test_linear_output_shape(self):
        """Test that linear layer produces correct output shape"""
        layer = Linear(3, 5)
        x = Tensor(np.random.randn(2, 3))  # Batch of 2
        y = layer(x)
        
        assert y.shape == (2, 5)
        
    def test_linear_computation(self):
        """Test linear layer computation"""
        layer = Linear(2, 1)
        
        # Set known weights
        layer.weight.data = np.array([[1.0], [2.0]], dtype=np.float32)
        layer.bias.data = np.array([0.5], dtype=np.float32)
        
        x = Tensor([[1.0, 1.0]])
        y = layer(x)
        
        # y = x @ W + b = [1, 1] @ [[1], [2]] + [0.5] = [3.5]
        assert np.allclose(y.data, [[3.5]])
        
    def test_linear_backward(self):
        """Test gradients flow through linear layer"""
        layer = Linear(2, 1)
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        
        y = layer(x)
        y.backward()
        
        # Gradients should be computed
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad is not None


class TestActivations:
    """Test activation functions"""
    
    def test_relu_forward(self):
        """Test ReLU forward pass"""
        relu = ReLU()
        x = Tensor([-2, -1, 0, 1, 2])
        y = relu(x)
        
        assert np.allclose(y.data, [0, 0, 0, 1, 2])
        
    def test_relu_backward(self):
        """Test ReLU backward pass"""
        relu = ReLU()
        x = Tensor([-1, 0, 1, 2], requires_grad=True)
        y = relu(x)
        y.backward(np.ones(4))
        
        assert np.allclose(x.grad, [0, 0, 1, 1])
        
    def test_log_softmax_forward(self):
        """Test LogSoftmax forward pass"""
        log_softmax = LogSoftmax(axis=1)
        x = Tensor([[1, 2, 3]])
        y = log_softmax(x)
        
        # Check that exp(log_softmax) sums to 1
        assert np.allclose(np.exp(y.data).sum(axis=1), 1.0)
        
    def test_log_softmax_backward(self):
        """Test LogSoftmax backward pass"""
        log_softmax = LogSoftmax(axis=1)
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = log_softmax(x)
        y.backward(np.ones((1, 3)))
        
        assert x.grad is not None
        assert x.grad.shape == (1, 3)


class TestComposedModel:
    """Test composed neural network models"""
    
    def test_simple_mlp(self):
        """Test a simple multi-layer perceptron"""
        class SimpleMLP(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(10, 5)
                self.relu = ReLU()
                self.fc2 = Linear(5, 2)
                
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        model = SimpleMLP()
        x = Tensor(np.random.randn(3, 10))  # Batch of 3
        y = model(x)
        
        assert y.shape == (3, 2)
        
    def test_mlp_parameters(self):
        """Test parameter collection in MLP"""
        class SimpleMLP(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(10, 5)
                self.fc2 = Linear(5, 2)
                
            def forward(self, x):
                return self.fc2(self.fc1(x))
        
        model = SimpleMLP()
        params = list(model.parameters())
        
        # Should have 4 parameters: 2 weights + 2 biases
        assert len(params) == 4
        
    def test_mlp_training_step(self):
        """Test a complete forward-backward pass"""
        class SimpleMLP(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(5, 3)
                self.relu = ReLU()
                self.fc2 = Linear(3, 2)
                
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        model = SimpleMLP()
        x = Tensor(np.random.randn(2, 5), requires_grad=True)
        
        # Forward
        y = model(x)
        loss = y.sum()
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        # Check gradients exist
        for p in model.parameters():
            assert p.grad is not None


class TestModelSaveLoad:
    """Test model save/load functionality"""
    
    def test_state_dict(self):
        """Test getting state dict from a model"""
        layer = Linear(3, 2)
        state = layer.state_dict()
        
        assert 'weight' in state
        assert 'bias' in state
        assert state['weight'].shape == (3, 2)
        assert state['bias'].shape == (2,)
        
    def test_load_state_dict(self):
        """Test loading state dict into a model"""
        layer1 = Linear(3, 2)
        layer2 = Linear(3, 2)
        
        # Get state from layer1
        state = layer1.state_dict()
        
        # Load into layer2
        layer2.load_state_dict(state)
        
        # Verify weights are the same
        assert np.allclose(layer1.weight.data, layer2.weight.data)
        assert np.allclose(layer1.bias.data, layer2.bias.data)
        
    def test_save_load(self):
        """Test saving and loading model to/from file"""
        import tempfile
        import os
        
        # Create a model
        layer = Linear(5, 3)
        original_weight = layer.weight.data.copy()
        original_bias = layer.bias.data.copy()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            layer.save(temp_path)
            
            # Create a new model and load
            new_layer = Linear(5, 3)
            new_layer.load(temp_path)
            
            # Verify weights are the same
            assert np.allclose(new_layer.weight.data, original_weight)
            assert np.allclose(new_layer.bias.data, original_bias)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def test_nested_module_save_load(self):
        """Test save/load with nested modules"""
        import tempfile
        import os
        
        class SimpleMLP(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(10, 5)
                self.fc2 = Linear(5, 2)
                
            def forward(self, x):
                return self.fc2(self.fc1(x))
        
        # Create and save model
        model = SimpleMLP()
        original_state = model.state_dict()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
            
        try:
            model.save(temp_path)
            
            # Create new model and load
            new_model = SimpleMLP()
            new_model.load(temp_path)
            new_state = new_model.state_dict()
            
            # Verify all parameters are the same
            for key in original_state:
                assert np.allclose(original_state[key], new_state[key])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def test_model_output_after_load(self):
        """Test that model produces same output after save/load"""
        import tempfile
        import os
        
        layer = Linear(3, 2)
        x = Tensor(np.random.randn(5, 3))
        
        # Get output before save
        output_before = layer(x).data.copy()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
            
        try:
            # Save and load
            layer.save(temp_path)
            new_layer = Linear(3, 2)
            new_layer.load(temp_path)
            
            # Get output after load
            output_after = new_layer(x).data
            
            # Outputs should be identical
            assert np.allclose(output_before, output_after)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
