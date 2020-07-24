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


from generalizedtrees.core import FeatureSpec
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


class SupervisedClassifierNode(ClassificationTreeNode):

    def __init__(self, data, targets, target_classes):
        super().__init__()
        self.src_data = data
        self.src_targets = targets
        self.target_classes = target_classes
        self.branch = None
        self.idx = None
        self.split: Optional[SplitTest] = None
    
    @property
    def data(self):
        return self.src_data[self.idx, :]
    
    @property
    def targets(self):
        return self.src_targets[self.idx]

    @cached_property
    def probabilities(self):
        slots = pd.Series(0, index=self.target_classes, dtype=np.float)
        freqs = pd.Series(self.targets).value_counts(normalize=True, sort=False)
        return slots.add(freqs, fill_value=0.0)

    def node_proba(self, data_matrix):
        n = data_matrix.shape[0]

        return pd.DataFrame([self.probabilities] * n)
    
    def __str__(self):
        if self.is_leaf:
            return f'Predict {dict(self.probabilities)}'
        else:
            return str(self.split)


def make_supervised_classifier_root(tree_model):
    node = SupervisedClassifierNode(tree_model.data, tree_model.targets, tree_model.target_classes)
    node.idx = np.arange(node.src_data.shape[0], dtype=np.intp)
    return node

def generate_supervised_classifier_children(tree_model, parent, split):
    # Note: this can be implemented without referencing tree_model or split.
    # Is that always the case?

    # Get branching for training samples
    branches = split.pick_branches(parent.data)

    for b in np.unique(branches):
        node = SupervisedClassifierNode(tree_model.data, tree_model.targets, tree_model.target_classes)
        node.idx = parent.idx[branches == b]
        node.branch = split.constraints[b]

        logger.debug(f'Created node with subview {node.idx}')
        yield node

def construct_supervised_classifier_split(tree_model, node: SupervisedClassifierNode):
    data = node.data
    targets = node.targets
    feature_spec = tree_model.feature_spec

    split_candidates = make_split_candidates(feature_spec, data, targets)
    best_split = null_split
    best_split_score = 0
    for split in split_candidates:
        new_score = information_gain(split, data, targets)
        if new_score > best_split_score:
            best_split_score = new_score
            best_split = split
    
    logger.debug(f'Best split "{best_split}" with score {best_split_score}')

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
    branches = split.pick_branches(data)
    return entropy(targets) - sum(map(
        lambda b: entropy(targets[branches == b]),
        np.unique(branches)))

