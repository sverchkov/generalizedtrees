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


from generalizedtrees.base import TreeEstimatorNode
from generalizedtrees.core import FeatureSpec
from generalizedtrees.splitting import fayyad_thresholds, one_vs_all
from generalizedtrees.scores import entropy
from functools import cached_property
from statistics import mode
from typing import List
import numpy as np
from logging import getLogger

logger = getLogger()

class SupervisedLearnerNode(TreeEstimatorNode):

    def __init__(self, data, targets):
        super().__init__()
        self.src_data = data
        self.src_targets = targets
        self.branch = None
        self.idx = None
    
    @property
    def data(self):
        return self.src_data[self.idx, :]
    
    @property
    def targets(self):
        return self.src_targets[self.idx]

    @cached_property
    def prediction(self):
        return mode(self.targets)
    
    def predict(self, data_vector):
        return self.prediction

    def pick_branch(self, data_vector):
    
        for i in range(len(self)):
            if self[i].branch.test(data_vector):
                return i
        
        logger.error(f"""\
            Unable to pick branch for:
            data vector: {data_vector}
            node: {self}""")
        raise ValueError

def make_supervised_learner_node(tree_model, branch=None, parent=None):

    node = SupervisedLearnerNode(tree_model.data, tree_model.targets)

    if branch is None or parent is None:
        node.idx = list(range(node.src_data.shape[0]))
    else:
        node.idx = [idx for idx in parent.idx if branch.test(node.src_data[idx, :])]
        node.branch = branch
        
    return node

def construct_supervised_learner_split(tree_model, node: SupervisedLearnerNode):
    data = node.data
    targets = node.targets
    feature_spec = tree_model.feature_spec

    split_candidates = make_split_candidates(feature_spec, data, targets)
    best_split = ()
    best_split_score = 0
    for split in split_candidates:
        new_score = information_gain(split, data, targets)
        if new_score > best_split_score:
            best_split_score = new_score
            best_split = split
    
    return best_split

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

def information_gain(split, data, targets):
    """
    Compute the split score (information gain) for a split.

    A split is a tuple of constraints.
    """
    return entropy(targets) - sum(map(
        lambda c: entropy(targets[np.apply_along_axis(c.test, 1, data)]),
        split))

