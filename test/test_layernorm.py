import numpy as np
from mytorch.autograd import Tensor
from mytorch.nn import LayerNorm


def test_layernorm_forward_shape():
    """Test that LayerNorm preserves input shape."""
    ln = LayerNorm(normalized_shape=4)  # Normalize over last dimension of size 4
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    
    out = ln(x)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"


def test_layernorm_normalizes_output():
    """Test that LayerNorm output has mean ≈ 0 and std ≈ 1 along normalized axis."""
    ln = LayerNorm(normalized_shape=8)
    # Use a tensor with known non-zero mean and non-unit variance
    x = Tensor(np.random.randn(4, 8).astype(np.float32) * 5 + 10)
    
    out = ln(x)
    
    # Check mean is approximately 0 along last axis
    out_mean = out.data.mean(axis=-1)
    assert np.allclose(out_mean, 0, atol=1e-5), f"Mean not zero: {out_mean}"
    
    # Check std is approximately 1 along last axis
    out_std = out.data.std(axis=-1)
    assert np.allclose(out_std, 1, atol=1e-5), f"Std not one: {out_std}"


def test_layernorm_weight_and_bias():
    """Test that weight (gamma) and bias (beta) are applied correctly."""
    ln = LayerNorm(normalized_shape=4)
    # Set custom weight and bias
    ln.weight = Tensor(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32), requires_grad=True)
    ln.bias = Tensor(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), requires_grad=True)
    
    x = Tensor(np.random.randn(2, 4).astype(np.float32) * 3 + 5)
    out = ln(x)
    
    # After normalization (mean=0, std=1), scaling by 2 and shifting by 1
    # should give mean ≈ 1 and std ≈ 2
    out_mean = out.data.mean(axis=-1)
    out_std = out.data.std(axis=-1)
    
    assert np.allclose(out_mean, 1, atol=1e-5), f"Mean not 1: {out_mean}"
    assert np.allclose(out_std, 2, atol=1e-5), f"Std not 2: {out_std}"


def test_layernorm_backward():
    """Test that gradients flow through LayerNorm."""
    ln = LayerNorm(normalized_shape=4)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    
    out = ln(x)
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Input gradient is None"
    assert ln.weight.grad is not None, "Weight gradient is None"
    assert ln.bias.grad is not None, "Bias gradient is None"
    
    # Check gradient shapes
    assert x.grad.shape == x.shape, f"Input gradient shape mismatch: {x.grad.shape}"
    assert ln.weight.grad.shape == ln.weight.shape, f"Weight gradient shape mismatch"
    assert ln.bias.grad.shape == ln.bias.shape, f"Bias gradient shape mismatch"


def test_layernorm_matches_numpy():
    """Test LayerNorm output matches manual numpy computation."""
    np.random.seed(42)
    ln = LayerNorm(normalized_shape=4, eps=1e-5)
    x_data = np.random.randn(3, 4).astype(np.float32)
    x = Tensor(x_data)
    
    out = ln(x)
    
    # Manual computation
    mean = x_data.mean(axis=-1, keepdims=True)
    var = x_data.var(axis=-1, keepdims=True)
    expected = (x_data - mean) / np.sqrt(var + 1e-5)
    # weight=1, bias=0 by default
    
    assert np.allclose(out.data, expected, atol=1e-5), f"Output mismatch:\n{out.data}\nvs\n{expected}"


def test_layernorm_3d_input():
    """Test LayerNorm works with 3D input (batch, seq, features)."""
    ln = LayerNorm(normalized_shape=8)
    x = Tensor(np.random.randn(2, 5, 8).astype(np.float32) * 2 + 3, requires_grad=True)
    
    out = ln(x)
    
    # Check normalization along last axis
    out_mean = out.data.mean(axis=-1)
    out_std = out.data.std(axis=-1)
    
    assert np.allclose(out_mean, 0, atol=1e-5), f"Mean not zero for 3D input"
    assert np.allclose(out_std, 1, atol=1e-5), f"Std not one for 3D input"


def test_layernorm_eps_prevents_division_by_zero():
    """Test that eps prevents division by zero for constant input."""
    ln = LayerNorm(normalized_shape=4, eps=1e-5)
    # Constant input has zero variance
    x = Tensor(np.ones((2, 4), dtype=np.float32) * 5)
    
    out = ln(x)
    
    # Should not have NaN or Inf
    assert not np.any(np.isnan(out.data)), "Output contains NaN"
    assert not np.any(np.isinf(out.data)), "Output contains Inf"


if __name__ == "__main__":
    test_layernorm_forward_shape()
    print("✓ test_layernorm_forward_shape passed")
    
    test_layernorm_normalizes_output()
    print("✓ test_layernorm_normalizes_output passed")
    
    test_layernorm_weight_and_bias()
    print("✓ test_layernorm_weight_and_bias passed")
    
    test_layernorm_backward()
    print("✓ test_layernorm_backward passed")
    
    test_layernorm_matches_numpy()
    print("✓ test_layernorm_matches_numpy passed")
    
    test_layernorm_3d_input()
    print("✓ test_layernorm_3d_input passed")
    
    test_layernorm_eps_prevents_division_by_zero()
    print("✓ test_layernorm_eps_prevents_division_by_zero passed")
    
    print("\nAll LayerNorm tests passed!")

