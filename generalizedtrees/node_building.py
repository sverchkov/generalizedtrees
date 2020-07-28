# Mixins for node-building in tree learners and supporting objects
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

from generalizedtrees.base import SplitTest, ClassificationTreeNode, TreeBuilder
from functools import cached_property
from typing import Optional
import numpy as np
import pandas as pd
from logging import getLogger

logger = getLogger()

# For supervised classifiers

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

class SupCferNodeBuilderMixin:

    def create_root(self):
        node = SupervisedClassifierNode(self.data, self.targets, self.target_classes)
        node.idx = np.arange(node.src_data.shape[0], dtype=np.intp)
        return node

    def generate_children(self, parent, split):
        # Note: this can be implemented without referencing tree_model or split.
        # Is that always the case?

        # Get branching for training samples
        branches = split.pick_branches(parent.data)

        for b in np.unique(branches):
            node = SupervisedClassifierNode(self.data, self.targets, self.target_classes)
            node.idx = parent.idx[branches == b]
            node.branch = split.constraints[b]

            logger.debug(f'Created node with subview {node.idx}')
            yield node