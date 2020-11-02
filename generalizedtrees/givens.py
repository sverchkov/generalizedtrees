# Givens learner components
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

from abc import abstractmethod
from typing import Callable, Optional, Tuple, Protocol
from generalizedtrees.features import FeatureSpec, infer_feature_spec
import pandas as pd
import numpy as np

# Givens Learner Component:

# Interface definition
class GivensLC(Protocol):
    """
    Givens Learner Component (LC)

    The givens learner component is used during fitting to process the inputs to the fit function
    and provide a standardized interface for acessing the 'givens', e.g. training data, oracle
    """

    feature_names: np.ndarray
    feature_spec: Tuple[FeatureSpec]
    target_names: np.ndarray
    data_matrix: np.ndarray
    target_matrix: np.ndarray

    @abstractmethod
    def process(self, *args, **kwargs) -> None:
        raise NotImplementedError


# For supervised data
class SupervisedDataGivensLC(GivensLC):
    """
    Givens LC for the standard supervised learning setting
    """

    #data_transform: Optional[Callable] = None # Maybe in the future
    transform_targets: Optional[Callable] = None

    def process(self, data, targets, *args, **kwargs) -> None:
        
        # Infer target_names if not given
        if 'target_names' not in kwargs:
            self.target_names = np.unique(targets)
        k = len(self.target_names)

        # Parse data and populate tree_builder
        self.data_matrix, self.feature_names, self.feature_spec = parse_data(
            data,
            feature_names=kwargs.get('feature_names'),
            feature_spec=kwargs.get('feature_spec')
        )
        
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        if self.transform_targets is not None:
            targets = self.transform_targets(targets)

        self.target_matrix = targets


# For unlabeled features + oracle


# Helpers

def parse_data(data, feature_names=None, feature_spec=None):

    data_shape = data.shape

    if len(data_shape) == 2:
        _, m = data_shape
    else:
        raise ValueError(f'Expected 2-dimensional data array buy data shape is {data_shape}')


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

    
    if feature_spec is None:
        raise NotImplementedError("Haven't implemented feature spec inference for this type yet")

    return (data_matrix, feature_names, feature_spec)


def fit_with_data_and_oracle(
    tree_builder: AbstractTreeBuilder,
    data,
    oracle,
    #oracle_gives_probabilities: bool = False, #TODO: Figure out need and scope
    #max_tree_size: Optional[int] = None, #TODO: Set in parameters/kwargs
    **kwargs):

    # So far just blindly accepting kwargs. A little hacky.
    tree_builder.__dict__.update(kwargs)

    # Parse data and populate tree_builder
    tree_builder.data, tree_builder.feature_names, tree_builder.feature_spec = parse_data(
        data,
        feature_names=getattr(tree_builder, 'feature_names', None),
        feature_spec=getattr(tree_builder, 'feature_spec', None)
    )

    # Do we use _d?
    _, tree_builder._d = np.shape(tree_builder.data)

    tree_builder.oracle = oracle

    tree_builder.oracle_gives_probabilities = getattr(tree_builder, 'oracle_gives_probabilities', True)

    # Targets of training data
    if tree_builder.oracle_gives_probabilities:
        target_proba = tree_builder.oracle(tree_builder.data)
        if isinstance(target_proba, pd.DataFrame):
            tree_builder.targets = target_proba.to_numpy()
            inferred_classes = target_proba.columns.to_numpy()
        else:
            # Add checks to verify shape/numpy-ness
            tree_builder.targets = target_proba
            inferred_classes = np.arange(target_proba.shape[1])
        
        tree_builder.classes_ = getattr(tree_builder, "classes_", inferred_classes)

    else:
        targets = tree_builder.oracle(tree_builder.data)
        if isinstance(targets, pd.Series):
            targets = targets.to_numpy()

        tree_builder.classes_ = getattr(tree_builder, "classes_", np.unique(targets))

        k = len(tree_builder.classes_)
        tree_builder.targets = np.zeros((len(targets), k))
        for i in range(k):
            tree_builder.targets[targets == tree_builder.classes_[i], i] = 1.0

    # Build the tree
    tree_builder.tree = tree_builder.build_tree()
    tree_builder.tree = tree_builder.prune_tree(tree_builder.tree)

    return tree_builder
