# Classes and functions implementing decision tree prediction
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from abc import abstractmethod
from generalizedtrees.givens import GivensLC
from typing import Protocol

import numpy as np

from generalizedtrees.tree import Tree


def _estimate_subtree(node: Tree.Node, data_matrix, idx, result_matrix):
    """
    Workhorse for estimation using a built tree.

    node: tree node,
        if it is a leaf, its item must contain a 'model' (Estimator) member;
        if it is internal, its item must contain a 'split' (SplitTest) member
    data_matrix: a numpy(-like) matrix of n instances by m features
    idx: a numpy array of numeric instance indexes
    result_matrix: a numpy matrix of outputs
    """

    if node.is_leaf:
        result_matrix[idx,:] = node.item.model.estimate(data_matrix[idx,:])
    
    else:
        branches = node.item.split.pick_branches(data_matrix[idx,:])
        for b in np.unique(branches):
            _estimate_subtree(node[b], data_matrix, idx[branches==b], result_matrix)

    return result_matrix


def estimate(tree: Tree, data_matrix, target_dimension):

    n = data_matrix.shape[0]

    return _estimate_subtree(
        tree.node('root'),
        data_matrix,
        np.arange(n, dtype=np.intp),
        np.empty(
            (n, target_dimension),
            dtype=np.float))


# Predictor Learner Component:

# Interface definition
class PredictorLC(Protocol):
    """
    Predictor Learner Component (LC)

    The predictor learner component computes a prediction for data given a learned tree.
    It correctly routes (and if needed transforms) the estimates obtained from the leaf estimators
    to outputs of the predict and predict_proba methods.

    It is assumed that set_target_names will be called to help determine the output shape before
    prediction is attempted. Typically, set_target_names will be called during fitting.
    """

    @abstractmethod
    def initialize(self, givens: GivensLC) -> 'PredictorLC':
        raise NotImplementedError

    def predict(self, tree: Tree, data_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, tree: Tree, data_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# Regressors
class RegressorLC(PredictorLC):
    """
    Regressor learner component

    Passes estimates from tree directly through the predict method.
    Watches the target_names attribute to get output dimension.
    """

    target_dim: int

    def initialize(self, givens: GivensLC) -> PredictorLC:
        self.target_dim = len(givens.target_names)

    def predict(self, tree: Tree, data_matrix: np.ndarray):
        return estimate(tree, data_matrix, self.target_dim)

# Classifiers

# Base for classifiers
class BaseClassifierLC(PredictorLC):
    """
    Base class for classifier LCs
    
    Defines set_target_names
    Defines predict as a function of predict_proba
    """

    target_names: np.ndarray

    def initialize(self, givens: GivensLC) -> PredictorLC:
        self.target_names = givens.target_names

    def predict(self, tree: Tree, data_matrix: np.ndarray):

        proba = self.predict_proba(tree, data_matrix)
        max_idx = proba.argmax(axis=1)
        return self.target_names[max_idx]

# Direct classifiers: each dimension of the estimator matches the probability of a target class
class ClassifierLC(BaseClassifierLC):
    """
    Classifier LC

    Each dimension of the underlying estimator matches the probability of a target class
    """

    def predict_proba(self, tree: Tree, data_matrix: np.ndarray):

        proba = estimate(tree, data_matrix, len(self.target_names))

        return proba


class BinaryClassifierLC(BaseClassifierLC):
    """
    Binary Classifier LC

    The binary classifier is different from the direct classifier in that the
    underlying estimator is assumed to be 1-dimensional, with the estimate reflecting
    the probability of the +ve (second) class
    """

    def predict_proba(self, tree: Tree, data_matrix: np.ndarray):

        p_1 = estimate(tree, data_matrix, 1)
        proba = np.concatenate((1 - p_1, p_1), axis=1)

        return proba
