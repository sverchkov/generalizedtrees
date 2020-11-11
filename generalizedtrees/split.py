# Split generation components
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

from logging import getLogger
from typing import Iterable, Protocol, Tuple, Optional
from functools import cached_property
from operator import itemgetter

import numpy as np

from generalizedtrees.base import SplitTest
from generalizedtrees.features import FeatureSpec
from generalizedtrees.constraints import LEQConstraint, GTConstraint, NEQConstraint, EQConstraint

logger = getLogger()

# Split classes

class SplitGT(SplitTest):

    def __init__(self, feature, value, feature_name=None):
        self.feature_name = feature_name
        self.feature = feature
        self.value = value
    
    def pick_branches(self, data_matrix):
        try:
            v = data_matrix[:, self.feature]
            return (v > self.value).astype(np.intp)
        except:
            logger.fatal(
                'Something went wrong.\n'
                f'Feature: {self.feature}, Value: {self.value}\n'
                f'Matrix:\n{data_matrix}\n'
                f'Column returned:\n{v}')
            raise

    @cached_property
    def constraints(self):
        return (
            LEQConstraint(self.feature, self.value),
            GTConstraint(self.feature, self.value))
    
    def __str__(self):
        if self.feature_name is None:
            feature = f'x[{self.feature}]'
        else:
            feature = self.feature_name
        return f'Test {feature} > {self.value}'


class SplitOneVsAll(SplitTest):

    def __init__(self, feature, value, feature_name=None):
        self.feature_name = feature_name
        self.feature = feature
        self.value = value
    
    def pick_branches(self, data_matrix):
        return (data_matrix[:, self.feature] == self.value).astype(np.intp)

    @cached_property
    def constraints(self):
        return (
            NEQConstraint(self.feature, self.value),
            EQConstraint(self.feature, self.value))

    def __str__(self):
        if self.feature_name is None:
            feature = f'x[{self.feature}]'
        else:
            feature = self.feature_name
        return f'Test {feature} == {self.value}'


class SplitEveryValue(SplitTest):

    def __init__(self, feature, values, feature_name=None):
        self.feature_name = feature_name
        self.feature = feature
        self.values = values
        self.map = {values[i]: i for i in range(len(values))}
    
    def pick_branches(self, data_matrix):
        return data_matrix[:, self.feature].map(self.map)

    @cached_property
    def constraints(self):
        return (EQConstraint(self.feature, v) for v in self.values)

    def __str__(self):
        if self.feature_name is None:
            feature = f'x[{self.feature}]'
        else:
            feature = self.feature_name
        return f'Test {feature} against each of {self.values}'


# Base split generating functions

def fayyad_thresholds(feature_vector, feature_index, target_matrix):
    """
    Generator of splits for numeric (or more generally orderable) values.

    We only generate splits between distinctly labeled distinct adjacent values.
    Fayyad and Irani (1992) prove that such splits are always better (in terms of entropy) than
    splits between adjacent equally labeled values.

    :param data: Input data feature vector, length n
    :param feature: Index of splitting feature (needed to create split object)
    :param target: Target value matrix, n-by-k
    """
    v = sorted(zip(feature_vector, target_matrix), key=itemgetter(0))
    
    # Flag for handling the case when two identical x-values have distinct y-values
    x_collision = False

    for j in range(1, len(v), 1):
        x_prev, y_prev = v[j-1]
        x, y = v[j]

        # Only place splits between distinct x-values
        if x_prev < x:
            # Only place splits between distinct y-values
            if x_collision or np.any(y_prev != y):
                split_point = (x_prev + x)/2
                yield SplitGT(feature_index, split_point)
            # Reset collision flag when advancing in x-space
            x_collision = False
        else:
            # Detect y-collision
            if np.any(y_prev != y):
                x_collision = True


def one_vs_all(feature_vector, feature_index):
    """
    Generator for one-vs-all splits.
    :param data: Input data matrix, n-by-d
    :param feature: Index of splitting feature (integer >=0 and <d)
    """

    values = np.unique(feature_vector)
    
    for x in values:
        yield SplitOneVsAll(feature_index, x)


def all_values_split(feature, values):
    """
    Generator for all-value splits.
    :param feature: Feature index
    :param values: List of possible feature values
    """
    yield SplitEveryValue(feature, values)


# Split Candidate Generators

# Interface definition
class SplitCandidateGeneratorLC(Protocol):

    def genenerator(self, data: np.ndarray, y: np.ndarray) -> Iterable[SplitTest]:
        raise NotImplementedError

# Default implementation for single-feature axis-aligned splits:
class AxisAlignedSplitGeneratorLC(SplitCandidateGeneratorLC):

    feature_spec: Tuple[FeatureSpec]

    def genenerator(self, data: np.ndarray, y: np.ndarray) -> Iterable[SplitTest]:

        for j in range(len(self.feature_spec)):
            if self.feature_spec[j] is FeatureSpec.CONTINUOUS:
                yield from fayyad_thresholds(data[:, j], j, y)
            elif self.feature_spec[j] & FeatureSpec.DISCRETE:
                yield from one_vs_all(data[:, j], j)
            else:
                raise ValueError(f"I don't know how to handle feature spec {self.feature_spec[j]}")
        
# Split Generator Learner Component

# Future: define protocol if we ever need to define multiple ways to do this

class SplitConstructorLC:

    split_generator: SplitCandidateGeneratorLC

    def construct_split(self, node, data: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> SplitTest:

        if data is None:
            data = node.data
        
        if y is None:
            y = node.model.estimate(data)
        
        best_split = None
        best_split_score = 0
        for split in self.split_generator.genenerator(data, y):
            new_score = self.score_split(node, split)
            if new_score > best_split_score:
                best_split_score = new_score
                best_split = split

        return best_split

    def score_split(self, node, split: SplitTest) -> float:

        # TODO: check what generalizes across existing implementations
        raise NotImplementedError

