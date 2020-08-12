# Definitions of splitting strategies for tree-building
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

import logging
import numpy as np
import pandas as pd
from functools import cached_property
from generalizedtrees.base import SplitTest
from generalizedtrees.constraints import LEQConstraint, GTConstraint, EQConstraint, NEQConstraint
from generalizedtrees.features import category_dtype

logger = logging.getLogger()

# Access helpers
def get_column(data, feature):
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, feature].array
    else:
        return data[:, feature]

# Split classes

class SplitGT(SplitTest):

    def __init__(self, feature, value, feature_name=None):
        self.feature_name = feature_name
        self.feature = feature
        self.value = value
    
    def pick_branches(self, data_matrix):
        try:
            v = get_column(data_matrix, self.feature)
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
        return (get_column(data_matrix, self.feature) == self.value).astype(np.intp)

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
        return get_column(data_matrix, self.feature).map(self.map)

    @cached_property
    def constraints(self):
        return (EQConstraint(self.feature, v) for v in self.values)

    def __str__(self):
        if self.feature_name is None:
            feature = f'x[{self.feature}]'
        else:
            feature = self.feature_name
        return f'Test {feature} against each of {self.values}'


# Test generators

def fayyad_thresholds(data, target, feature):
    """
    Generator of splits for numeric (or more generally orderable) values.

    We only generate splits between distinctly labeled distinct adjacent values.
    Fayyad and Irani (1992) prove that such splits are always better (in terms of entropy) than
    splits between adjacent equally labeled values.

    :param data: Input data matrix, n-by-d
    :param target: Target value vector of length n
    :param feature: Index of splitting feature (integer >=0 and <d)
    """
    if isinstance(data, pd.DataFrame):
        v = sorted(zip(data.iloc[:,feature], target))
        feature_name = data.columns[feature]
    else:
        v = sorted(zip(data[:,feature], target))
        feature_name = None

    # Flag for handling the case when two identical x-values have distinct y-values
    x_collision = False

    for j in range(1, len(v), 1):
        x_prev, y_prev = v[j-1]
        x, y = v[j]

        # Only place splits between distinct x-values
        if x_prev < x:
            # Only place splits between distinct y-values
            if x_collision or y_prev != y:
                split_point = (x_prev + x)/2
                yield SplitGT(feature, split_point, feature_name=feature_name)
            # Reset collision flag when advancing in x-space
            x_collision = False
        else:
            # Detect y-collision
            if y_prev != y:
                x_collision = True


def fayyad_thresholds_p(data: pd.DataFrame, target_proba: pd.DataFrame, feature: int):
    """
    Generator of splits for numeric (or more generally orderable) values.

    We only generate splits between distinctly labeled distinct adjacent values.
    This is an adaptation of the original method (for hard labels) to probability vectors.
    Fayyad and Irani (1992) prove that such splits are always better (in terms of entropy) than
    splits between adjacent equally labeled values.

    :param data: Input data frame, n-by-d
    :param target_proba: Target probability data frame, n-by-k
    :param feature: Index of splitting feature (integer >=0 and <d)
    """
    if isinstance(data, pd.DataFrame):
        v = sorted(zip(data.iloc[:,feature], target_proba.itertuples(index=False)))
        feature_name = data.columns[feature]
    else:
        v = sorted(zip(data[:,feature], target_proba.itertuples(index=False)))
        feature_name = None

    # Flag for handling the case when two identical x-values have distinct y-values
    x_collision = False

    for j in range(1, len(v), 1):
        x_prev, y_prev = v[j-1]
        x, y = v[j]
        
        # Only place splits between distinct x-values
        if x_prev < x:
            # Only place splits between distinct y-values
            if x_collision or any(a != b for a, b in zip(y_prev, y)):
                split_point = (x_prev + x)/2
                yield SplitGT(feature, split_point, feature_name=feature_name)
            # Reset collision flag when advancing in x-space
            x_collision = False
        else:
            # Detect x-collision
            if y_prev != y:
                x_collision = True


def one_vs_all(data, feature):
    """
    Generator for one-vs-all splits.
    :param data: Input data matrix, n-by-d
    :param feature: Index of splitting feature (integer >=0 and <d)
    """

    if isinstance(data, pd.DataFrame):
        column = data.iloc[:,feature]
        if category_dtype == column.dtype:
            values = column.cat.categories
        else:
            values = column.unique()
        
        feature_name = data.columns[feature]
    else:
        values = np.unique(data[:,feature])
        feature_name = None
    
    for x in values:
        yield SplitOneVsAll(feature, x, feature_name=feature_name)


def binary_threshold(data, feature):
    """
    Deprecated
    """
    for x_i in data:
        yield [LEQConstraint(feature, x_i[feature]), GTConstraint(feature, x_i[feature])]


def all_values_split(feature, values):
    """
    Generator for all-value splits.
    :param feature: Feature index
    :param values: List of possible feature values
    """
    yield SplitEveryValue(feature, values)


def compose_splitting_strategies(spec_list):
    """
    Translates a list of specification into a working splitting strategy function
    :param spec_list:
    :return:
    """
    raise NotImplementedError
