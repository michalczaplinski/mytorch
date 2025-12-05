import numpy as np
from mytorch.autograd import Tensor


def test_reshape_forward():
    """Test that reshape correctly changes tensor shape."""
    x = Tensor(np.arange(12).reshape(3, 4), requires_grad=True)

    out = x.reshape(4, 3)

    assert out.shape == (4, 3), f"Shape mismatch: {out.shape}"
    assert np.allclose(out.data, x.data.reshape(4, 3)), "Data mismatch after reshape"


def test_reshape_backward():
    """Test that gradients are correctly reshaped back to original shape."""
    x = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)

    out = x.reshape(4, 3)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradient is None"
    assert x.grad.shape == (3, 4), f"Gradient shape mismatch: {x.grad.shape}"
    assert np.allclose(x.grad, np.ones((3, 4))), "Gradient values incorrect"


def test_reshape_preserves_data_order():
    """Test that reshape preserves element order (row-major/C order)."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = Tensor(data, requires_grad=True)

    out = x.reshape(6)

    expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    assert np.allclose(out.data, expected), f"Data order mismatch: {out.data}"


def test_transpose_forward():
    """Test that transpose correctly reorders axes."""
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)

    out = x.transpose(2, 0, 1)  # (2,3,4) -> (4,2,3)

    assert out.shape == (4, 2, 3), f"Shape mismatch: {out.shape}"
    assert np.allclose(out.data, x.data.transpose(2, 0, 1)), (
        "Data mismatch after transpose"
    )


def test_transpose_backward():
    """Test that gradients are correctly transposed back."""
    x = Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), requires_grad=True)

    out = x.transpose(2, 0, 1)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradient is None"
    assert x.grad.shape == (2, 3, 4), f"Gradient shape mismatch: {x.grad.shape}"
    assert np.allclose(x.grad, np.ones((2, 3, 4))), "Gradient values incorrect"


def test_transpose_2d_swap():
    """Test simple 2D matrix transpose (swap rows and columns)."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # (2, 3)
    x = Tensor(data, requires_grad=True)

    out = x.transpose(1, 0)  # (2,3) -> (3,2)

    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    assert out.shape == (3, 2), f"Shape mismatch: {out.shape}"
    assert np.allclose(out.data, expected), f"Data mismatch: {out.data}"


def test_transpose_backward_gradient_flow():
    """Test that gradients flow correctly through transpose with non-trivial loss."""
    x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), requires_grad=True)

    # Transpose then multiply by weights to create asymmetric gradients
    out = x.transpose(1, 0)  # [[1,3], [2,4]]
    weights = Tensor(np.array([[1, 0], [0, 2]], dtype=np.float32))
    weighted = out * weights  # [[1,0], [0,8]]
    loss = weighted.sum()
    loss.backward()

    # Gradient at out: [[1,0], [0,2]] (from weights)
    # Transpose back: [[1,0], [0,2]].T = [[1,0], [0,2]] (symmetric in this case)
    expected_grad = np.array([[1, 0], [0, 2]], dtype=np.float32)
    assert x.grad is not None, "Gradient is None"
    assert np.allclose(x.grad, expected_grad), f"Gradient mismatch: {x.grad}"
