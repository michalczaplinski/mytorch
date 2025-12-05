import numpy as np

from mytorch.autograd import Tensor


def test_transpose_backward_2d_swap():
    """Test 2D transpose (swap axes) - this should work even with the bug."""
    np.random.seed(42)
    x_data = np.random.randn(3, 4).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)

    # Forward: (3, 4) -> (4, 3) with axes (1, 0)
    y = x.transpose(1, 0)
    assert y.shape == (4, 3), f"Expected shape (4, 3), got {y.shape}"

    # Backward with gradient of ones
    y.backward(np.ones_like(y.data))

    # Gradient should have same shape as x
    assert x.grad is not None, "Gradient is None"
    assert x.grad.shape == x.shape, (
        f"Gradient shape {x.grad.shape} doesn't match input shape {x.shape}"
    )

    # For a simple sum loss, gradient should be all ones
    expected_grad = np.ones_like(x_data)
    np.testing.assert_allclose(x.grad, expected_grad, rtol=1e-5)
    print("✓ 2D swap transpose backward PASSED")


def test_transpose_backward_3d_rotation():
    """Test 3D transpose with rotation (2, 0, 1) - this reveals the bug."""
    np.random.seed(42)
    x_data = np.random.randn(2, 3, 4).astype(np.float32)  # shape (2, 3, 4)
    x = Tensor(x_data, requires_grad=True)

    # Forward: (2, 3, 4) -> (4, 2, 3) with axes (2, 0, 1)
    # axes (2, 0, 1) means: new_dim_0 = old_dim_2, new_dim_1 = old_dim_0, new_dim_2 = old_dim_1
    y = x.transpose(2, 0, 1)
    assert y.shape == (4, 2, 3), f"Expected shape (4, 2, 3), got {y.shape}"

    # Backward with gradient of ones
    y.backward(np.ones_like(y.data))

    # Gradient should have same shape as x
    assert x.grad is not None, "Gradient is None"
    assert x.grad.shape == x.shape, (
        f"Gradient shape {x.grad.shape} doesn't match input shape {x.shape}"
    )

    # For a simple sum loss, gradient should be all ones
    expected_grad = np.ones_like(x_data)
    np.testing.assert_allclose(x.grad, expected_grad, rtol=1e-5)
    print("✓ 3D rotation transpose backward PASSED")


def test_transpose_backward_numerical_gradient():
    """Numerical gradient check for transpose with non-trivial permutation."""
    np.random.seed(42)
    x_data = np.random.randn(2, 3, 4).astype(
        np.float64
    )  # Use float64 for better precision
    axes = (2, 0, 1)
    eps = 1e-5

    # Compute analytical gradient
    x = Tensor(x_data.copy(), requires_grad=True)
    y = x.transpose(*axes)
    # Use sum as loss function
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Gradient is None"
    analytical_grad = x.grad.copy()

    # Compute numerical gradient
    numerical_grad = np.zeros_like(x_data)
    for idx in np.ndindex(x_data.shape):
        x_plus = x_data.copy()
        x_plus[idx] += eps
        loss_plus = np.sum(x_plus.transpose(axes))

        x_minus = x_data.copy()
        x_minus[idx] -= eps
        loss_minus = np.sum(x_minus.transpose(axes))

        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4)
    print("✓ Numerical gradient check PASSED")


def test_transpose_backward_with_weighted_loss():
    """Test with a non-uniform gradient to catch subtle bugs."""
    np.random.seed(42)
    x_data = np.random.randn(2, 3, 4).astype(
        np.float64
    )  # Use float64 for better precision
    grad_out = np.random.randn(4, 2, 3).astype(np.float64)  # Random upstream gradient
    axes = (2, 0, 1)
    eps = 1e-5

    # Compute analytical gradient
    x = Tensor(x_data.copy(), requires_grad=True)
    y = x.transpose(*axes)
    y.backward(grad_out)
    assert x.grad is not None, "Gradient is None"
    analytical_grad = x.grad.copy()

    # Compute numerical gradient
    # Loss = sum(grad_out * y) = sum(grad_out * x.transpose(axes))
    numerical_grad = np.zeros_like(x_data)
    for idx in np.ndindex(x_data.shape):
        x_plus = x_data.copy()
        x_plus[idx] += eps
        loss_plus = np.sum(grad_out * x_plus.transpose(axes))

        x_minus = x_data.copy()
        x_minus[idx] -= eps
        loss_minus = np.sum(grad_out * x_minus.transpose(axes))

        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4)
    print("✓ Weighted loss numerical gradient check PASSED")
