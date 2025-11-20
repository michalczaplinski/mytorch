# train_mnist.py

import numpy as np
import struct
import sys
from pathlib import Path

import os
# Ensure project root is on sys.path when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import our framework components
from mytorch.autograd import Tensor
from mytorch.nn import Module, Linear, ReLU, LogSoftmax, nll_loss
from mytorch.optim import SGD

# --- 1. Model Definition ---

class SimpleMLP(Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)
        self.log_softmax = LogSoftmax(axis=1) # Apply softmax along class dimension

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

# --- 2. Data Loading (NumPy-only) ---

def fetch_mnist(data_path: str = ".") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses the MNIST files in raw ubyte format (not gzipped).
    """
    data_dir = Path(data_path)

    print(data_dir)
    
    def parse_images(path: Path | str) -> np.ndarray:
        with open(path, 'rb') as f:
            # Read header: Magic(4), NumImages(4), Rows(4), Cols(4)
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid MNIST image file magic number: {magic}")
            # Read data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, rows * cols)

    def parse_labels(path: Path | str) -> np.ndarray:
        with open(path, 'rb') as f:
            # Read header: Magic(4), NumItems(4)
            magic, _ = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid MNIST label file magic number: {magic}")
            # Read data
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = parse_images(data_dir / "train-images.idx3-ubyte")
    y_train = parse_labels(data_dir / "train-labels.idx1-ubyte")
    X_test = parse_images(data_dir / "t10k-images.idx3-ubyte")
    y_test = parse_labels(data_dir / "t10k-labels.idx1-ubyte")
    
    # Normalize and convert types
    return (X_train.astype(np.float32) / 255.0, y_train.astype(np.int64),
            X_test.astype(np.float32) / 255.0, y_test.astype(np.int64))

# --- 3. Main Training Script ---

if __name__ == "__main__":
    
    # --- Config ---
    INPUT_SIZE = 784  # 28x28
    HIDDEN_SIZE = 64
    NUM_CLASSES = 10
    LR = 0.001
    BATCH_SIZE = 64
    EPOCHS = 50
    
    DATA_DIR = "./examples/mnist/data/"
    
    # --- Load Data ---
    print(f"Loading MNIST data from {DATA_DIR}...")
    try:
        X_train, y_train, X_test, y_test = fetch_mnist(DATA_DIR)
        print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Could not load MNIST data.")
        sys.exit(1)
    
    # --- Initialize Model and Optimizer ---
    model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
    optimizer = SGD(model.parameters(), lr=LR)
    
    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        num_batches = X_train.shape[0] // BATCH_SIZE
        total_loss = 0.0
        
        # Shuffle data
        perm = np.random.permutation(X_train.shape[0])
        
        for i in range(num_batches):
            # Get batch
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            idx = perm[start:end]
            
            X_batch = Tensor(X_train[idx])
            y_batch_targets = y_train[idx] # Targets are just numpy arrays

            # 1. Zero gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            predictions = model(X_batch)
            
            # 3. Compute loss
            loss = nll_loss(predictions, y_batch_targets)
            total_loss += loss.data.item()
            
            # 4. Backward pass
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

    # --- Evaluation ---
    print("Training complete. Evaluating on test set...")
    
    # No need for gradients during evaluation
    test_preds_tensor = model(Tensor(X_test, requires_grad=False))
    
    # Get class predictions (argmax of log_probs)
    pred_labels = np.argmax(test_preds_tensor.data, axis=1)
    
    # Compute accuracy as the mean of correct predictions
    correct_predictions = (pred_labels == y_test).astype(np.float32)
    accuracy: float = correct_predictions.mean().item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")