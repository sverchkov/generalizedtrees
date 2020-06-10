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
from generalizedtrees.constraints import LEQConstraint, GTConstraint, EQConstraint, NEQConstraint

logger = logging.getLogger()


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
    v = sorted(zip(data[:,feature], target))

    for j in range(1, len(v), 1):
        x_prev, y_prev = v[j-1]
        x, y = v[j]

        # Flag for handling the case when two identical x-values have distinct y-values
        y_collision = False

        # Only place splits between distinct x-values
        if x_prev < x:
            # Only place splits between distinct y-values
            if y_collision or y_prev != y:
                split_point = (x_prev + x)/2
                yield (LEQConstraint(feature, split_point), GTConstraint(feature, split_point))
            # Reset collision flag when advancing in x-space
            y_collision = False
        else:
            # Detect y-collision
            if y_prev != y:
                y_collision = True


def one_vs_all(data, feature):
    """
    Generator for one-vs-all splits.
    :param data: Input data matrix, n-by-d
    :param feature: Index of splitting feature (integer >=0 and <d)
    """
    for x in np.unique(data[:,feature]):
        yield [EQConstraint(feature, x), NEQConstraint(feature, x)]


def binary_threshold(data, feature):
    for x_i in data:
        yield [LEQConstraint(feature, x_i[feature]), GTConstraint(feature, x_i[feature])]


def all_values_split(feature, values):
    yield [EQConstraint(feature, value) for value in values]


def compose_splitting_strategies(spec_list):
    """
    Translates a list of specification into a working splitting strategy function
    :param spec_list:
    :return:
    """
    raise NotImplementedError
