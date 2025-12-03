import numpy as np
import sys


# Import utils
from utils import fetch_mnist

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