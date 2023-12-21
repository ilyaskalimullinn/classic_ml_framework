import abc
import numpy as np

class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X: np.ndarray, *args, **kwargs) -> None:
        raise NotImplementedError
