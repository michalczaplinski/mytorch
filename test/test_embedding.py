import numpy as np
from mytorch.autograd import Tensor
from mytorch.nn import Embedding


def test_embedding_forward():
    """Test that forward pass correctly indexes into the weight matrix."""
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    indices = Tensor(np.array([0, 2, 1]))
    
    out = emb(indices)
    
    expected = emb.weight.data[[0, 2, 1]]
    assert np.allclose(out.data, expected), f"Forward mismatch: {out.data} vs {expected}"


def test_embedding_backward():
    """Test that gradients accumulate at the correct indices."""
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    indices = Tensor(np.array([0, 2, 1]))
    
    out = emb(indices)
    loss = out.sum()
    loss.backward()
    
    expected_grad = np.zeros((5, 3), dtype=np.float32)
    expected_grad[[0, 2, 1]] = 1.0
    assert emb.weight.grad is not None, "Gradient is None"
    assert np.allclose(emb.weight.grad, expected_grad), f"Gradient mismatch: {emb.weight.grad}"


def test_embedding_repeated_indices():
    """Test that repeated indices correctly accumulate gradients."""
    emb = Embedding(num_embeddings=4, embedding_dim=2)
    indices = Tensor(np.array([1, 1, 1, 0]))  # Index 1 appears 3 times
    
    out = emb(indices)
    loss = out.sum()
    loss.backward()
    
    expected_grad = np.zeros((4, 2), dtype=np.float32)
    expected_grad[0] = 1.0  # Index 0 appears once
    expected_grad[1] = 3.0  # Index 1 appears 3 times
    assert emb.weight.grad is not None, "Gradient is None"
    assert np.allclose(emb.weight.grad, expected_grad), f"Repeated index gradient mismatch: {emb.weight.grad}"


if __name__ == "__main__":
    test_embedding_forward()
    print("✓ test_embedding_forward passed")
    
    test_embedding_backward()
    print("✓ test_embedding_backward passed")
    
    test_embedding_repeated_indices()
    print("✓ test_embedding_repeated_indices passed")
    
    print("\nAll embedding tests passed!")

