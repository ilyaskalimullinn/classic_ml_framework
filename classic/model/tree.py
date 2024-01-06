from __future__ import annotations
import abc
from typing import Any, Literal, Self, Tuple, Union, Optional
import numpy as np

from classic.model.base import BaseModel


class Node:
    def __init__(
        self,
        left_child: Optional[Self] = None,
        right_child: Optional[Self] = None,
        split_ind: Optional[int] = None,
        split_val: Optional[float] = None,
        terminal_node: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        self.left_child = left_child
        self.right_child = right_child
        self.split_ind = split_ind
        self.split_val = split_val
        self.terminal_node = terminal_node


class DecisionTree(BaseModel):
    d: int
    subsample_size: int
    root: Node

    def __init__(
        self,
        criterion: str,
        max_depth: int = 0,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
        feature_subsample_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample_ratio = feature_subsample_ratio
        self.random_state = random_state

    def _calc_node_criterion(
        self, targets: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> float:
        return getattr(self, f"_calc_node_criterion_{self.criterion}")(targets, weights)

    def fit(
        self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.d = X.shape[1]
        self.subsample_size = int(np.ceil(self.d * self.feature_subsample_ratio))
        self.root = Node()
        self._build_splitting_node(self.root, X, y, 0, weights)

    def _build_splitting_node(
        self,
        node: Node,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        if (self.max_depth != 0 and depth == self.max_depth) or (
            X.shape[0] < self.min_samples_split
        ):
            self._build_terminal_node(node, X, y, weights)
            return

        axes = self._get_axes()
        best_criterion = 1e300

        for feature_name in axes:
            feature_vals = np.unique(X[:, feature_name])
            feature_vals = np.sort(feature_vals)
            for feature_val in feature_vals[
                :-1
            ]:  # no reason to check if values are greater than greatest value
                criterion_expectation = self._calc_split_criterion(
                    X, y, feature_name, feature_val, weights
                )
                if criterion_expectation < best_criterion:
                    best_criterion = criterion_expectation
                    node.split_ind = feature_name
                    node.split_val = feature_val

        # split to fit node's children
        X_left, X_right, y_left, y_right, weights_left, weights_right = self._split(
            X, y, node.split_ind, node.split_val, weights
        )

        node.left_child = Node()
        node.right_child = Node()
        self._build_splitting_node(
            node.left_child, X_left, y_left, depth + 1, weights_left
        )
        self._build_splitting_node(
            node.right_child, X_right, y_right, depth + 1, weights_right
        )

    def _get_axes(self) -> np.ndarray:
        a = np.arange(self.d)
        np.random.shuffle(a)
        return a[: self.subsample_size]

    def _calc_split_criterion(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_name: int,
        feature_val: float,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        _, __, y_left, y_right, weights_left, weights_right = self._split(
            X, y, feature_name, feature_val, weights
        )

        left_criterion = self._calc_node_criterion(y_left, weights_left)
        right_criterion = self._calc_node_criterion(y_right, weights_right)
        criterion = (
            y_left.size * left_criterion + y_right.size * right_criterion
        ) / y.size
        return criterion

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_name: int,
        feature_val: float,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        is_greater_mask = X[:, feature_name] > feature_val
        is_lesser_mask = ~is_greater_mask
        weights_left, weights_right = None, None
        if weights is not None:
            weights_left = weights[is_lesser_mask]
            weights_right = weights[is_greater_mask]

        X_left = X[is_lesser_mask]
        X_right = X[is_greater_mask]

        y_left = y[is_lesser_mask]
        y_right = y[is_greater_mask]

        return X_left, X_right, y_left, y_right, weights_left, weights_right

    @abc.abstractmethod
    def _build_terminal_node(
        self,
        node: Node,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        pass

    def predict(self, X: np.ndarray) -> None:
        pred = []
        for i in range(X.shape[0]):
            pred.append(self._predict_one(X[i]))

        return np.vstack(pred)

    def _predict_one(self, x: np.ndarray) -> Any:
        node = self.root
        while node.terminal_node is None:
            node = (
                node.right_child
                if x[node.split_ind] > node.split_val
                else node.left_child
            )
        return node.terminal_node


class DecisionTreeRegressor(DecisionTree):
    def __init__(
        self,
        criterion: Literal["variance"] = "variance",
        max_depth: int = 0,
        min_samples_split: int = 2,
        random_state: int | None = None,
        subsample_ratio: float = 1,
    ) -> None:
        super().__init__(
            criterion, max_depth, min_samples_split, random_state, subsample_ratio
        )

    def _calc_node_criterion_variance(
        self, targets: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        if weights is None:
            return np.var(targets)
        else:
            a = np.square(targets - np.mean(targets))
            a = a * weights
            return np.mean(a)

    def _build_terminal_node(
        self,
        node: Node,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        if weights is None:
            node.terminal_node = np.mean(y)
        else:
            node.terminal_node = np.sum(weights * y) / np.sum(weights)


class DecisionTreeClassifier(DecisionTree):
    def __init__(
        self,
        num_classes: int,
        max_depth: int = 0,
        criterion: Literal["entropy"] = "entropy",
        min_samples_split: int = 2,
        random_state: int | None = None,
        subsample_ratio: float = 1,
        min_class_proba: float = 1e-3,
    ) -> None:
        super().__init__(
            criterion, max_depth, min_samples_split, random_state, subsample_ratio
        )
        self.num_classes = num_classes
        self.min_class_proba = min_class_proba

    def _calc_node_criterion_entropy(
        self, targets: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        if weights is None:
            p = np.unique(targets, return_counts=True)[1]
            p = p / targets.shape[0]
            res = -np.sum(p * np.log2(p))
            return res
        else:
            # todo maybe change???
            p = np.zeros(k)
            for k in range(self.num_classes):
                p[k] = weights[targets == k]

            p[p <= 0] = self.min_class_proba
            p /= np.sum(p)
            res = -np.sum(p * np.log2(p))
            return res

    def _build_terminal_node(
        self,
        node: Node,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        if weights is None:
            p = np.bincount(y, minlength=self.num_classes)
            p = p / y.shape[0]
        else:
            p = np.zeros(k)
            for k in range(self.num_classes):
                p[k] = weights[y == k]

            p[p <= 0] = self.min_class_proba
            p /= np.sum(p)

        node.terminal_node = p  # .reshape(1, -1)


class DecisionTreeMultiRegressor(DecisionTree):
    def __init__(
        self,
        criterion: Literal["variance"] = "variance",
        max_depth: int = 0,
        min_samples_split: int = 2,
        random_state: int | None = None,
        subsample_ratio: float = 1,
    ) -> None:
        super().__init__(
            criterion, max_depth, min_samples_split, random_state, subsample_ratio
        )

    def _calc_node_criterion_variance(
        self, targets: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        if weights is None:
            return np.var(targets, axis=0).mean()
        else:
            a = np.square(targets - np.mean(targets, axis=0))
            a = a * weights.reshape(-1, 1)
            a = a.mean(axis=0)
            return np.mean(a)

    def _build_terminal_node(
        self,
        node: Node,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        if weights is None:
            node.terminal_node = np.mean(y, axis=0)
        else:
            node.terminal_node = np.sum(y * weights.reshape(-1, 1)) / np.sum(weights)
