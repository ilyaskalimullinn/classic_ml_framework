import abc
from typing import List, Literal, Optional

import numpy as np
from scipy.special import softmax
from classic.model.base import BaseModel
from classic.model.tree import (
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeMultiRegressor,
    DecisionTreeRegressor,
)
from classic.util import one_hot_encode


class RandomForest(BaseModel):
    def __init__(
        self,
        criterion: str,
        n_estimators: int = 50,
        max_depth: int = 0,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
        colsample_bynode: float = 1.0,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.colsample_bynode = colsample_bynode
        self.trees = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> None:
        if self.random_state:
            np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            tree = self._create_tree()
            tree.fit(X, y, weights)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> None:
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        predictions = np.array(predictions)
        return predictions.mean(axis=0)

    @abc.abstractmethod
    def _create_tree(self) -> DecisionTree:
        pass


class RandomForestRegressor(RandomForest):
    def __init__(
        self,
        criterion: Literal["variance"] = "variance",
        n_estimators: int = 50,
        max_depth: int = 0,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
        colsample_bynode: float = 1,
    ) -> None:
        super().__init__(
            criterion,
            n_estimators,
            max_depth,
            min_samples_split,
            random_state,
            colsample_bynode,
        )

    def _create_tree(self) -> DecisionTree:
        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=None,
            subsample_ratio=self.colsample_bynode,
        )


class RandomForestClassifier(RandomForest):
    def __init__(
        self,
        num_classes: int,
        criterion: Literal["entropy"] = "entropy",
        n_estimators: int = 50,
        max_depth: int = 0,
        min_samples_split: int = 2,
        random_state: int | None = None,
        colsample_bynode: float = 1,
        min_class_proba: float = 1e-3,
    ) -> None:
        super().__init__(
            criterion,
            n_estimators,
            max_depth,
            min_samples_split,
            random_state,
            colsample_bynode,
        )
        self.num_classes = num_classes
        self.min_class_proba = min_class_proba

    def _create_tree(self) -> DecisionTree:
        return DecisionTreeClassifier(
            num_classes=self.num_classes,
            max_depth=self.max_depth,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            random_state=None,
            subsample_ratio=self.colsample_bynode,
            min_class_proba=self.min_class_proba,
        )


class GradientBoosting(BaseModel):
    estimators: List[BaseModel]

    def __init__(
        self,
        criterion: str,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        allow_weights=True,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.allow_weights = allow_weights
        self.estimators = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> None:
        if weights is not None and not self.allow_weights:
            raise ValueError("This model does not allow weights")

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        estimator_first = self._create_estimator()
        self._fit_estimator(estimator_first, X, y, weights)
        self.estimators.append(estimator_first)
        predictions = self.learning_rate * estimator_first.predict(X)

        for i in range(self.n_estimators - 1):
            residuals = -self._calc_criterion_gradient(y, predictions)
            estimator = self._create_estimator()
            self._fit_estimator(estimator, X, residuals, weights)
            self.estimators.append(estimator)
            predictions += self.learning_rate * estimator.predict(X)

    def _calc_criterion_gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        return getattr(self, f"_calc_criterion_gradient_{self.criterion}")(
            y_true, y_pred
        )

    @abc.abstractmethod
    def _create_estimator(self) -> BaseModel:
        pass

    def _fit_estimator(
        self,
        estimator: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        if self.allow_weights:
            estimator.fit(X, y, weights)
        else:
            estimator.fit(X, y)

    def _predict_weighted_sum(self, X: np.ndarray) -> np.ndarray:
        pred = 0
        for estimator in self.estimators:
            pred += estimator.predict(X)
        return pred * self.learning_rate

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_weighted_sum(X)


class GradientBoostedDecisionTrees(GradientBoosting):
    def __init__(
        self,
        criterion: str,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: int | None = None,
        colsample_bynode: float = 0.7,
    ) -> None:
        super().__init__(
            criterion,
            n_estimators,
            learning_rate,
            allow_weights=True,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.colsample_bynode = colsample_bynode


class GradientBoostingRegressor(GradientBoostedDecisionTrees):
    def __init__(
        self,
        criterion: Literal["squared_error"] = "squared_error",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: int | None = None,
        colsample_bynode: float = 0.7,
    ) -> None:
        super().__init__(
            criterion,
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_split,
            random_state,
            colsample_bynode,
        )

    def _calc_criterion_gradient_squared_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        return y_pred - y_true

    def _create_estimator(self) -> DecisionTree:
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            subsample_ratio=self.colsample_bynode,
        )


class GradientBoostingClassifier(GradientBoostedDecisionTrees):
    def __init__(
        self,
        num_classes: int,
        criterion: Literal["cross_entropy"] = "cross_entropy",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
        colsample_bynode: float = 0.7,
    ) -> None:
        super().__init__(
            criterion,
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_split,
            random_state,
            colsample_bynode,
        )
        self.num_classes = num_classes
        self.softmax = softmax

    def _calc_criterion_gradient_cross_entropy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        # y_pred are logits, y_true is one-hot-encoded
        return self.softmax(y_pred, axis=1) - y_true

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_weighted_sum(X)
        return self.softmax(logits, axis=1)

    def fit(
        self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> None:
        y_ohe = one_hot_encode(y, self.num_classes)
        return super().fit(X, y_ohe, weights)

    def _create_estimator(self) -> DecisionTree:
        return DecisionTreeMultiRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            subsample_ratio=self.colsample_bynode,
        )
