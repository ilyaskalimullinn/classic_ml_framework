import numpy as np

def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.zeros(shape=(y.size, num_classes))
    result[np.arange(y.size), y] = 1
    return result
