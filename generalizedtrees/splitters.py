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


from generalizedtrees.features import FeatureSpec
from generalizedtrees.base import ClassificationTreeNode, SplitTest, null_split
from generalizedtrees.splitting import fayyad_thresholds, one_vs_all
from generalizedtrees.scores import entropy
from functools import cached_property
from statistics import mode
from typing import List, Optional
import numpy as np
import pandas as pd
from logging import getLogger

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
