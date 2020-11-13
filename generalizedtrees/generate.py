# Unlabeled data generation
#
# Copyright 2020 Yuriy Sverchkov
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

from typing import Collection, Iterable, Optional, Protocol
from abc import abstractmethod
from logging import getLogger

import numpy as np
from scipy.stats import ks_2samp, chisquare

from generalizedtrees.constraints import Constraint, test
from generalizedtrees.features import FeatureSpec

logger = getLogger()

# Utility functions

def same_distribution(data_1, data_2, /, feature_spec, alpha):
    """
    Performs statistical test to determine if data are from the
    same distribution.

    data_1 and data_2 need to be numpy matrices
    """
    n_tests = 0
    min_p = 1.0
    for i in range(len(feature_spec)):

        if feature_spec[i] & FeatureSpec.DISCRETE:
            # Get frequencies.
            # Note: we're assuming that the union of values present in the samples is
            # the set of possible values. This is not all possible values that the
            # variable could originally take.
            v1, c1 = np.unique(data_1[:, i], return_counts=True)
            map1 = {v1[j]: c1[j] for j in range(len(v1))}

            v2, c2 = np.unique(data_2[:, i], return_counts=True)
            map2 = {v2[j]: c2[j] for j in range(len(v2))}

            values = np.union1d(v1, v2)
            k = len(values)

            # If only one value is present skip this test
            if k > 1:

                freq1 = [map1[v] if v in map1 else 0 for v in values]
                freq2 = [map2[v] if v in map2 else 0 for v in values]

                _, p = chisquare(f_obs=freq1, f_exp=freq2, ddof=k-1)
                if p < min_p:
                    min_p = p
                n_tests += 1
        
        else:
            # KS-test
            _, p = ks_2samp(data_1[:, i], data_2[:, i])
            if p < min_p:
                min_p = p
            n_tests += 1

    return min_p < alpha/n_tests


##################################
# Data Factory Learner Component #
##################################


# Interface definition
class DataFactoryLC(Protocol):

    feature_spec: Collection[FeatureSpec]

    def refit(self, data_matrix: np.ndarray) -> 'DataFactoryLC':
        return self

    @abstractmethod
    def generate(self, n: int, constraints: Iterable[Constraint] = ()) -> np.ndarray:
        raise NotImplementedError


# Implementations
class TrepanDataFactoryLC(DataFactoryLC):

    data_matrix: Optional[np.ndarray] = None
    alpha: float

    max_attemts: int
    on_timeout: str = 'partial'
    rng: np.random.Generator

    def __init__(
        self,
        alpha: float = 0.05,
        max_attempts: int = 1000,
        on_timeout: str = 'partial',
        rng: np.random.Generator = np.random.default_rng()) -> None:

        self.alpha = alpha
        self.max_attemts = max_attempts
        self.on_timeout = on_timeout
        self.rng = rng


    def refit(self, data_matrix: np.ndarray) -> 'DataFactoryLC':

        if (self.data_matrix is not None and
            same_distribution(data_matrix, self.data_matrix, feature_spec=self.feature_spec, alpha = self.alpha)
        ):
            return self
        
        self.data_matrix = data_matrix
    
    def generate(self, n: int, constraints: Iterable[Constraint] = ()) -> np.ndarray:

        n = max(0, n)

        logger.debug(f'Drawing sample of size {n}')

        result = np.empty((n, self.data_matrix.shape[1]))

        if n < 1:
            return result
        
        else:

            i = 0
            oversample_prop = 1

            for _ in range(self.max_attempts):
                needed = n - i
                n_sampled = round(min(needed * oversample_prop, self.max_sample))
                sample = self._generate(n_sampled)
                accepted = test(constraints, sample)
                n_acc = sum(accepted)
                n_fit = min(n_acc, needed)
                logger.debug(
                    f'Sampling loop: i={i}, n={n}. '
                    f'Needed {needed} samples, sampled {n_sampled}, accepted {n_acc}, fit {n_fit}')
                result[i:(i+n_fit),:] = sample[accepted,:][0:n_fit,:]
                i += n_fit
                if i >= n:
                    return result
                oversample_prop = max(oversample_prop, n_sampled / n_acc) if n_acc > 0 else np.inf
            
            logger.critical('Could not generate an acceptable sample within a reasonable time.')
            logger.debug(f'Failed to generate a sample for constraints: {constraints}')
            if self.on_timeout == 'partial':
                logger.warning('Returning partial sample')
                return result[1:i,:]
            elif self.on_timeout == 'dirty':
                logger.warning('Matrix with uninitialized elements')
                return result
            else: # on_timeout == 'raise':
                raise RuntimeError('Could not generate an acceptable sample within a reasonable time.')
    
    def _generate(self, n: int) -> np.ndarray:

        # The Trepan generator independently generates the individual feature values.
        return np.column_stack([self._generate_feature(i, n) for i in range(self.data_matrix.shape[1])])
    
    def _generate_feature(self, i: int, n: int) -> np.ndarray:

        if self.feature_spec[i] is FeatureSpec.CONTINUOUS:
            # Sample from a KDE.
            # We use Generator and not RandomState but KDE implementations use RandomState
            # so it's more reliable to just implement the sampling here. 
            return self.rng.normal(
                loc = self.rng.choice(self.data_matrix[:, i], size=n),
                scale = 1/np.sqrt(self.data_matrix.shape[0]),
                size = n)[0]

        elif self.feature_spec[i] & FeatureSpec.DISCRETE:
            # Sample from the empirical distribution
            values, counts = np.unique(self.data_matrix[:,i], return_counts=True)
            return self.rng.choice(values, p=counts/self.data_matrix.shape[0], size=n)
        
        else:
            raise ValueError(f"I don't know how to handle feature spec {self.feature_spec[i]}")

