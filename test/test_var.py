import numpy as np
from mytorch.autograd import Tensor


def test_var_forward_full():
    """Test variance over all elements."""
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = Tensor(x_data)
    
    out = x.var()
    
    expected = np.var(x_data)
    assert np.allclose(out.data, expected), f"Var mismatch: {out.data} vs {expected}"


def test_var_forward_axis0():
    """Test variance along axis 0."""
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = Tensor(x_data)
    
    out = x.var(axis=0)
    
    expected = np.var(x_data, axis=0)
    assert out.shape == expected.shape, f"Shape mismatch: {out.shape} vs {expected.shape}"
    assert np.allclose(out.data, expected), f"Var mismatch: {out.data} vs {expected}"


def test_var_forward_axis1():
    """Test variance along axis 1 (last axis)."""
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = Tensor(x_data)
    
    out = x.var(axis=1)
    
    expected = np.var(x_data, axis=1)
    assert out.shape == expected.shape, f"Shape mismatch: {out.shape} vs {expected.shape}"
    assert np.allclose(out.data, expected), f"Var mismatch: {out.data} vs {expected}"


def test_var_forward_axis_negative():
    """Test variance along axis -1 (last axis)."""
    x_data = np.random.randn(2, 3, 4).astype(np.float32)
    x = Tensor(x_data)
    
    out = x.var(axis=-1)
    
    expected = np.var(x_data, axis=-1)
    assert out.shape == expected.shape, f"Shape mismatch: {out.shape} vs {expected.shape}"
    assert np.allclose(out.data, expected), f"Var mismatch: {out.data} vs {expected}"


def test_var_forward_keepdims():
    """Test variance with keepdims=True."""
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = Tensor(x_data)
    
    out = x.var(axis=1, keepdims=True)
    
    expected = np.var(x_data, axis=1, keepdims=True)
    assert out.shape == expected.shape, f"Shape mismatch: {out.shape} vs {expected.shape}"
    assert np.allclose(out.data, expected), f"Var mismatch: {out.data} vs {expected}"


def test_var_backward_exists():
    """Test that backward pass produces gradients."""
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True)
    
    out = x.var()
    out.backward()
    
    assert x.grad is not None, "Gradient is None"
    assert x.grad.shape == x.shape, f"Gradient shape mismatch: {x.grad.shape}"


def test_var_backward_axis():
    """Test backward pass with axis parameter."""
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True)
    
    out = x.var(axis=1)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient is None"
    assert x.grad.shape == x.shape, f"Gradient shape mismatch: {x.grad.shape}"


def test_var_backward_numerical():
    """Test backward pass against numerical gradient."""
    np.random.seed(42)
    x_data = np.random.randn(3, 4).astype(np.float32)
    x = Tensor(x_data.copy(), requires_grad=True)
    
    # Forward and backward
    out = x.var(axis=-1)
    loss = out.sum()
    loss.backward()
    
    # Numerical gradient
    eps = 1e-4
    numerical_grad = np.zeros_like(x_data)
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_plus = x_data.copy()
            x_plus[i, j] += eps
            x_minus = x_data.copy()
            x_minus[i, j] -= eps
            
            loss_plus = np.var(x_plus, axis=-1).sum()
            loss_minus = np.var(x_minus, axis=-1).sum()
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    assert x.grad is not None, "Gradient is None"
    # Use rtol=1e-2 for float32 numerical gradient precision
    assert np.allclose(x.grad, numerical_grad, rtol=1e-2, atol=1e-5), \
        f"Gradient mismatch:\nAnalytic:\n{x.grad}\nNumerical:\n{numerical_grad}"


def test_var_zero_variance():
    """Test variance of constant values (should be 0)."""
    x = Tensor(np.ones((2, 4), dtype=np.float32) * 5)
    
    out = x.var(axis=-1)
    
    expected = np.zeros(2, dtype=np.float32)
    assert np.allclose(out.data, expected), f"Var of constant should be 0: {out.data}"


if __name__ == "__main__":
    test_var_forward_full()
    print("✓ test_var_forward_full passed")
    
    test_var_forward_axis0()
    print("✓ test_var_forward_axis0 passed")
    
    test_var_forward_axis1()
    print("✓ test_var_forward_axis1 passed")
    
    test_var_forward_axis_negative()
    print("✓ test_var_forward_axis_negative passed")
    
    test_var_forward_keepdims()
    print("✓ test_var_forward_keepdims passed")
    
    test_var_backward_exists()
    print("✓ test_var_backward_exists passed")
    
    test_var_backward_axis()
    print("✓ test_var_backward_axis passed")
    
    test_var_backward_numerical()
    print("✓ test_var_backward_numerical passed")
    
    test_var_zero_variance()
    print("✓ test_var_zero_variance passed")
    
    print("\nAll var() tests passed!")

