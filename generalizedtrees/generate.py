# Unlabeled data generation
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from typing import Collection, Iterable, Optional, Protocol
from abc import abstractmethod
from logging import getLogger

import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

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

                freq = [[map1[v] if v in map1 else 0 for v in values],
                        [map2[v] if v in map2 else 0 for v in values]]

                _, p, _, _ = chi2_contingency(observed=freq)

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


# Base classes
class ConstraintFreeSamplingFactoryLC(DataFactoryLC):

    data_matrix: Optional[np.ndarray] = None

    max_attempts: int
    max_sample: int
    on_timeout: str = 'partial'

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

                #if not all(accepted):
                #    logger.debug(f'Constraints: {constraints}')
                #    for j in np.nonzero(accepted):
                #        logger.debug(f'Rejected sample: {sample[j,:]}')
                
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


    @abstractmethod
    def _generate(self, n: int) -> np.ndarray:
        raise NotImplementedError


# Implementations
class TrepanDataFactoryLC(ConstraintFreeSamplingFactoryLC):

    alpha: float

    rng: np.random.Generator

    def __init__(
        self,
        alpha: float = 0.05,
        max_attempts: int = 1000,
        max_sample: int = 100000,
        on_timeout: str = 'partial',
        rng: np.random.Generator = np.random.default_rng()) -> None:

        self.alpha = alpha
        self.max_attempts = max_attempts
        self.max_sample = max_sample
        self.on_timeout = on_timeout
        self.rng = rng


    def copy(self) -> 'TrepanDataFactoryLC':
        """
        Make a deep copy
        """
        clone = TrepanDataFactoryLC(
            alpha=self.alpha,
            max_attempts=self.max_attempts,
            max_sample=self.max_sample,
            on_timeout=self.on_timeout,
            rng=self.rng)
        clone.feature_spec = self.feature_spec

        return clone


    def refit(self, data_matrix: np.ndarray) -> 'DataFactoryLC':

        if len(data_matrix) < 1 or (
            self.data_matrix is not None and
            same_distribution(data_matrix, self.data_matrix, feature_spec=self.feature_spec, alpha = self.alpha)
        ):
            return self
        
        clone = self.copy()
        clone.data_matrix = data_matrix

        return clone
        
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
                size = n)

        elif self.feature_spec[i] & FeatureSpec.DISCRETE:
            # Sample from the empirical distribution
            values, counts = np.unique(self.data_matrix[:,i], return_counts=True)
            return self.rng.choice(values, p=counts/self.data_matrix.shape[0], size=n)
        
        else:
            raise ValueError(f"I don't know how to handle feature spec {self.feature_spec[i]}")


class SmearingDataFactoryLC(ConstraintFreeSamplingFactoryLC):

    p_alt: float
    rng: np.random.Generator

    def __init__(
        self,
        p_alt: float = 0.5,
        max_attempts: int = 1000,
        max_sample: int = 100000,
        on_timeout: str = 'partial',
        rng: np.random.Generator = np.random.default_rng()
    ) -> None:

        self.p_alt = p_alt
        self.max_attempts = max_attempts
        self.max_sample = max_sample
        self.on_timeout = on_timeout
        self.rng = rng

    def refit(self, data_matrix: np.ndarray) -> DataFactoryLC:
        if self.data_matrix is None:
            self.data_matrix = data_matrix
        
        return self

    def _generate(self, n) -> np.ndarray:

        m, d = self.data_matrix.shape

        matrix_a = self.data_matrix[self.rng.integers(m, size=n, dtype=np.intp),:]
        matrix_b = self.data_matrix[self.rng.integers(m, size=n, dtype=np.intp),:]
        replacemask = self.rng.choice([False, True], size=n*d, p=[1-self.p_alt, self.p_alt]).reshape(n,d)
        matrix_a[replacemask] = matrix_b[replacemask]

        return matrix_a       
