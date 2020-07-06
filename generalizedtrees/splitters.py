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
from typing import List
import numpy as np


class SupervisedScoreSplitter:

    def __init__(self):
        super.__init__()
        self.data: np.ndarray
        self.targets: np.ndarray
        self.feature_spec: List[FeatureSpec]

    class Node(TreeEstimatorNode):

        def __init__(self):
            super().__init__()
            self.training_idx = None

    def new_node(self, branch=None, parent=None):

        node = SupervisedScoreSplitter.Node

        if branch is None or parent is None:
            node.training_idx = list(range(self.data.shape[0]))
        
        else:
            node.training_idx = [idx for idx in parent.training_idx if branch.test(self.data[idx, :])]
        
        return node

    def construct_split(self, node: SupervisedScoreSplitter.Node):
        data = self.data[node.training_idx, :]
        targets = self.targets[node.training_idx]

        split_candidates = self.make_split_candidates(data, targets)
        best_split = ()
        best_split_score = 0
        for split in split_candidates:
            new_score = self.split_score(split, data, targets)
            if new_score > best_split_score:
                best_split_score = new_score
                best_split = split
        
        return best_split

    def make_split_candidates(self, data, targets):
        # Note: we could make splits based on original training examples only, or on
        # training examples and generated examples.
        # In the current form, this could be controlled by the calling function.

        result = []

        for j in range(len(self.feature_spec)):
            if self.feature_spec[j] is FeatureSpec.CONTINUOUS:
                result.extend(fayyad_thresholds(data, targets, j))
            elif self.feature_spec[j] & FeatureSpec.DISCRETE:
                result.extend(one_vs_all(data, j))
            else:
                raise ValueError(f"I don't know how to handle feature spec {self.feature_spec[j]}")
        
        return result

    def split_score(self, split, data, targets):
        """
        Compute the split score (information gain) for a split.
        """
        return entropy(targets) - sum(map(
            lambda c: entropy(targets[np.apply_along_axis(c.test, 1, data)]),
            split))

