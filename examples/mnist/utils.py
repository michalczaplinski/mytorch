import numpy as np
import struct
from pathlib import Path

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
