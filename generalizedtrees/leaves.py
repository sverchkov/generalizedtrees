# Leaf models for trees
#
# Copyright 2019-2020 Yuriy Sverchkov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import Protocol


class LocalEstimator(Protocol):

    def fit(self, x, y, **kwargs) -> 'LocalEstimator':
        return self

    def estimate(self, data_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConstantEstimator(LocalEstimator):

    def __init__(self):
        self.est_vector: np.ndarray
    
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


class SKProbaEstimator(LocalEstimator):
    """
    Wrapper around an sklearn classifier, uses the predict_proba method.
    """

    def __init__(self, classifier):
        self.classifier = classifier
    
    def fit(self, x, y, **kwargs):
        self.classifier.fit(x, y, **kwargs)
        return self

    def estimate(self, data_matrix: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(data_matrix)