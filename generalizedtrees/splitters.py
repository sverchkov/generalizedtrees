# Splitter functions used in tree-building
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


import numpy as np
from logging import getLogger
from generalizedtrees.scores import entropy, entropy_of_p_matrix
from generalizedtrees.splitting import fayyad_thresholds, fayyad_thresholds_p, one_vs_all
from generalizedtrees.features import FeatureSpec

logger = getLogger()


def make_split_candidates(feature_spec, data, targets):
    # Note: we could make splits based on original training examples only, or on
    # training examples and generated examples.
    # In the current form, this could be controlled by the calling function.

    result = []

    for j in range(len(feature_spec)):
        if feature_spec[j] is FeatureSpec.CONTINUOUS:
            result.extend(fayyad_thresholds(data, targets, j))
        elif feature_spec[j] & FeatureSpec.DISCRETE:
            result.extend(one_vs_all(data, j))
        else:
            raise ValueError(f"I don't know how to handle feature spec {feature_spec[j]}")
    
    return result


def make_split_candidates_p(feature_spec, data, target_proba):

    result = []

    for j in range(len(feature_spec)):
        if feature_spec[j] is FeatureSpec.CONTINUOUS:
            result.extend(fayyad_thresholds_p(data, target_proba, j))
        elif feature_spec[j] & FeatureSpec.DISCRETE:
            result.extend(one_vs_all(data, j))
        else:
            raise ValueError(f"I don't know how to handle feature spec {feature_spec[j]}")
    
    return result


# Score functions for splits

def information_gain(split, data, targets):
    """
    Compute the split score (information gain) for a split.

    A split is a tuple of constraints.
    """
    branches = split.pick_branches(data)
    return entropy(targets) - sum(map(
        lambda b: entropy(targets[branches == b]),
        np.unique(branches)))

def information_gain_p(split, data, target_proba):
    """
    Compute the split score (information gain) for a split.

    A split is a tuple of constraints.

    This form uses probability estimates for targets.
    """
    branches = split.pick_branches(data)

    return entropy_of_p_matrix(target_proba) - sum(map(
        lambda b: entropy_of_p_matrix(target_proba[branches==b]),
        np.unique(branches)
    ))
