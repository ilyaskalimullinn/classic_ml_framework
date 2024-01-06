import abc
from typing import Literal
import numpy as np
from classic.exception import NotFittedError
from classic.model.base import BaseModel
import sklearn
from scipy.special import expit

from classic.util.scheduler import BaseScheduler, StaticScheduler


class BaseLinearRegression(BaseModel):
    def __init__(self, fit_intercept: bool = True) -> None:
        super().__init__()
        self.weights = None
        self.fit_intercept = fit_intercept
    
    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> None:
        assert y.shape[0] == X.shape[0], f"X and y must have the same number of elements, got X shape: {X.shape}, y shape: {y.shape}"
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if self.fit_intercept:
            X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        self._fit(X, y, *args, **kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise NotFittedError
        
        if self.fit_intercept:
            X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        
        return self._predict(X)

    def _predict(self, X_with_ones: np.ndarray) -> np.ndarray:
        return X_with_ones @ self.weights
    
    @abc.abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> None:
        raise NotImplementedError
    

class LinearRegression(BaseLinearRegression):
    def __init__(self, fit_intercept: bool = True) -> None:
        super().__init__(fit_intercept)
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        inverse_matrix = self._calc_pseudo_inverse_matrix(X)
        self.weights = inverse_matrix @ y
    
    def _calc_pseudo_inverse_matrix(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.inv(X.T @ X) @ X.T


class RidgeRegression(BaseLinearRegression):
    def __init__(self, fit_intercept: bool = True, alpha: float = 0) -> None:
        super().__init__(fit_intercept)
        self.alpha = alpha

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        inverse_matrix = self._calc_pseudo_inverse_matrix_ident(X)
        self.weights = inverse_matrix @ y

    def _calc_pseudo_inverse_matrix_ident(self, X: np.ndarray) -> np.ndarray:
        ident = np.identity(X.shape[1])
        if self.fit_intercept:
            ident[0][0] = 0
        return np.linalg.inv(X.T @ X + self.alpha * ident) @ X.T

class SGDRegression(BaseLinearRegression):
    def __init__(self, 
                 fit_intercept: bool = True, 
                 loss: Literal["mse"] = "mse",
                 lr_scheduler: BaseScheduler = StaticScheduler(),
                 n_epochs: int = 1000,
                 batch_size: int = 1,
                 do_shuffle: bool = True,
                 regularisation: Literal["l2"] = "l2",
                 alpha: float = 0) -> None:
        super().__init__(fit_intercept)
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.regularisation = regularisation
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if do_shuffle:
            self.shuffle = sklearn.utils.shuffle

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._init_weights(X, y)
        
        n_elements = X.shape[0]
        n_batches = n_elements // self.batch_size
        for epoch in range(1, self.n_epochs + 1):
            self.lr_scheduler.step(epoch, self.n_epochs)
            if self.shuffle:
                X, y = self.shuffle(X, y)
            for batch in range(0, n_batches):
                start = batch * self.batch_size
                end = start + self.batch_size
                grad = self._calc_grad(X[start:end], y[start:end])
                self.weights -= self.lr_scheduler.get_learning_rate() * grad

    def _calc_grad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        grad = getattr(self, f"_calc_grad_loss_{self.loss}")(X, y)
        grad /= X.shape[0]
        grad += getattr(self, f"_calc_grad_reg_{self.regularisation}")(X, y)
        return grad
    
    def _calc_grad_loss_mse(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return X.T @ (X @ self.weights - y)
    
    def _calc_grad_reg_l2(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.copy(self.weights)
        if self.fit_intercept:
            weights[0] = 0
        return self.alpha * weights

    def _init_weights(self, X_with_ones: np.ndarray, y: np.ndarray) -> None:
        d = X_with_ones.shape[1]
        std = (2 / d) ** 0.5
        self.weights = np.random.normal(loc=0, scale=std, size=(d, 1))


class LinearBinaryClassificator(SGDRegression):
    def __init__(self, 
                 fit_intercept: bool = True, 
                 loss: Literal['log', 'hinge'] = "log", 
                 lr_scheduler: BaseScheduler = StaticScheduler(), 
                 n_epochs: int = 1000, 
                 batch_size: int = 1, 
                 do_shuffle: bool = True, 
                 regularisation: Literal['l2'] = "l2", 
                 alpha: float = 0) -> None:
        super().__init__(fit_intercept, loss, lr_scheduler, n_epochs, 
                         batch_size, do_shuffle, regularisation, alpha)
        self.sigmoid = expit

    def _predict(self, X_with_ones: np.ndarray) -> np.ndarray:
        pred = super()._predict(X_with_ones)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred
    
    def _calc_grad_loss_log(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        grad = - y * self.sigmoid( - (X @ self.weights) * y)
        grad = X.T @ grad
        return grad


class LogisticRegression(LinearBinaryClassificator):
    def __init__(self, 
                 fit_intercept: bool = True,
                 lr_scheduler: BaseScheduler = StaticScheduler(), 
                 n_epochs: int = 1000, 
                 batch_size: int = 1, 
                 do_shuffle: bool = True, 
                 regularisation: Literal['l2'] = "l2", 
                 alpha: float = 0) -> None:
        super().__init__(fit_intercept, "log", lr_scheduler, n_epochs, 
                         batch_size, do_shuffle, regularisation, alpha)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        
        logits = X @ self.weights
        return expit(logits)
