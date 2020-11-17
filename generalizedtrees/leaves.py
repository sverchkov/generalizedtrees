# Leaf models for trees
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

import numpy as np
from typing import Any, Protocol


class LocalEstimator(Protocol):

    def fit(self, x, y, **kwargs) -> 'LocalEstimator':
        return self

    def estimate(self, data_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConstantEstimator(LocalEstimator):

    est_vector: np.ndarray
    
    def fit(self, x=None, y=None, **kwargs):

        if y is None:
            assert x is not None
            target_matrix = x
        else:
            target_matrix = y
        
        self.est_vector = target_matrix.mean(axis=0)

        return self

    def estimate(self, data_matrix: np.ndarray) -> np.ndarray:

        assert data_matrix.ndim == 2
        
        return np.repeat(np.reshape(self.est_vector, (1,-1)), data_matrix.shape[0], axis=0)
    
    def __str__(self):
        return str(self.est_vector)


class SKProbaClassifier(LocalEstimator):
    """
    Wrapper around an sklearn classifier.
    
    To predict, uses the predict_proba method.
    To fit, converts the y matrix to a label vectors, selecting the maximal component for each instance.
    """

    classifier: Any # Use monkey-typing
    fallback: bool = True # Whether to fall back to constant estimator when fitting data with one class.

    def __init__(self, classifier):
        self.classifier = classifier
    
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):

        # Record number of classes
        self.classifier.classes_ = np.arange(y.shape[1])

        # Convert y matrix to label vector
        targets = y.argmax(axis=1)

        if self.fallback and len(np.unique(targets)) < 2:
            return ConstantEstimator().fit(y)

        self.classifier.fit(x, targets, **kwargs)
        return self

    def estimate(self, data_matrix: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(data_matrix)
