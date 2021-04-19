# Givens learner components
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from abc import abstractmethod
from typing import Callable, Optional, Tuple, Protocol
from logging import getLogger

import pandas as pd
import numpy as np
from generalizedtrees.features import FeatureSpec, infer_feature_spec

logger = getLogger()

# Givens Learner Component:

# Interface definition
class GivensLC(Protocol):
    """
    Givens Learner Component (LC)

    The givens learner component is used during fitting to process the inputs to the fit function
    and provide a standardized interface for acessing the 'givens', e.g. training data, oracle
    """

    # "Output" properties
    feature_names: np.ndarray
    feature_spec: Tuple[FeatureSpec]
    target_names: Optional[np.ndarray] = None
    data_matrix: np.ndarray
    target_matrix: Optional[np.ndarray] = None
    oracle: Optional[Callable[..., np.ndarray]] = None

    @abstractmethod
    def process(self, *args, **kwargs) -> None:
        raise NotImplementedError


# For supervised data
class SupervisedDataGivensLC(GivensLC):
    """
    Givens LC for the standard supervised learning setting
    """

    def __init__(self, binary_classification: bool = False) -> None:
        """
        :param: binary_classification tells us to internally model only the probability of the +ve (second) class
        """
        self.binary_classification = binary_classification

    def process(
        self,
        data,
        targets,
        *args,
        target_names = None,
        target_shape = None,
        **kwargs) -> None:
        """
        Processing input for standard supervised learning:

        :param: data - the n-by-d feature matrix
        :param: targets - either a vector of target values of length n or an n-by-k matrix
        :param: target_shape - a hint to which shape the targets take: 'label_vector' for
            a vector of length n where values correspond to class labels, or 'matrix' for
            an n-by-k matrix. If None, then this is inferred from the dimensionality of targets.
        """

        if target_names is not None:
            self.target_names = target_names

        if target_shape is None:
            target_shape = 'matrix' if len(targets.shape) == 2 else 'label_vector'
        
        # Infer target_names if not given
        if self.target_names is None:
            if target_shape == 'label_vector':
                self.target_names = np.unique(targets)
            else:
                # Assume matrix with 1 column per target
                self.target_names = getattr(targets, 'columns', np.arange(targets.shape[1]))

        # Parse data
        self.data_matrix, self.feature_names, self.feature_spec = parse_data(
            data,
            feature_names=kwargs.get('feature_names'),
            feature_spec=kwargs.get('feature_spec')
        )
        
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        # Reshape targets to matrix form
        if target_shape == 'label_vector':
            # This does 1-hot encoding
            self.target_matrix = np.array([x == self.target_names for x in targets], dtype=float)
        else:
            self.target_matrix = targets

        # Only keep the 2nd column for binary classification
        if self.binary_classification:
            self.target_matrix = self.target_matrix[:,[1]]


# For unlabeled features + oracle
class DataWithOracleGivensLC(GivensLC):
    """
    Given LC for the usual explanation setting, where we have unlabeled training data and
    a predictor oracle.
    """

    def process(self, data, oracle, *args, target_names=None, prelabel_data=True, feature_groups=None, **kwargs):
        """
        Processing input for the explanation setting:

        :param: data - the n-by-d feature matrix
        :param: oracle - the oracle, can either be a SKLearn-like classifier, in which case the predict function is
            used, or a function that takes data and outputs a prediction (in matrix form).
        :param: target_names - The names of the target classes/components.
        :param: prelabel_data - Whether to run the oracle on the data first, before building the model
        :param: feature_groups - Used in some explanation configurations - A list of lists of feature indeces
            representing semantically meaningful feature groups
        """

        if target_names is not None:
            self.target_names = target_names
        
        if feature_groups is not None:
            # TODO: Validation
            self.feature_groups = feature_groups

        # Parse data
        self.data_matrix, self.feature_names, self.feature_spec = parse_data(
            data,
            feature_names=kwargs.get('feature_names'),
            feature_spec=kwargs.get('feature_spec')
        )

        # Infer correct oracle
        if hasattr(oracle, 'predict') and hasattr(oracle, 'classes_'):
            logger.info(
                'Inferring that oracle is a Scikit-Learn-like classifier '
                'and using the "predict" method.')
            self.oracle = lambda x: np.eye(len(oracle.classes_))[oracle.predict(x).astype('intp'),]
        else:
            logger.info('Treating oracle as a function')
            self.oracle = oracle

        if prelabel_data:
            targets = self.oracle(self.data_matrix)

            # Infer target_names if not given
            if self.target_names is None:
                # Assume that oracle yields matrix or dataframe
                self.target_names = getattr(targets, 'columns', np.arange(targets.shape[1]))
            
            # Assume that oracle yields matrix
            self.target_matrix = targets if isinstance(targets, np.ndarray) else targets.to_numpy()


# Helpers

def parse_data(data, feature_names=None, feature_spec=None):

    data_shape = data.shape

    if len(data_shape) == 2:
        _, m = data_shape
    else:
        raise ValueError(f'Expected 2-dimensional data array buy data shape is {data_shape}')

    data_matrix = None

    if isinstance(data, pd.DataFrame):
        if feature_names is None:
            feature_names = data.columns
        if feature_spec is None:
            feature_spec = infer_feature_spec(data)
        
        # TODO: Convert non-numerics appropriately to integers
        data_matrix = data.to_numpy()

    if feature_names is not None:

        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.flatten()
        
        else:
            feature_names = np.array(feature_names)

        if len(feature_names) != m:
            raise ValueError(f'Got feature names of length {len(feature_names)}, but expected {m} features.')

    else:

        feature_names = np.arange(m)

    if isinstance(data, np.ndarray):
        data_matrix = data
    
    if data_matrix is None:
        raise ValueError(f'Could not process a {type(data)} object: {data}')
    
    if feature_spec is None:
        logger.warning('Assuming continuous features in the absence of feature specifications')
        feature_spec = (FeatureSpec.CONTINUOUS,) * m

    return (data_matrix, feature_names, feature_spec)

