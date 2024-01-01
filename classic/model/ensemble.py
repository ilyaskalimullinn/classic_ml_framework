import abc
from typing import Literal, Optional

import numpy as np
from classic.model.base import BaseModel
from classic.model.tree import (
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


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

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
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
