import numpy as np

def one_hot_encode_grid_labels(grid_labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[grid_labels]
