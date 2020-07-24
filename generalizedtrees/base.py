# Base class hierarchy for tree models
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

from abc import ABC, abstractmethod
from generalizedtrees.tree import tree_to_str, Tree, TreeNode
from typing import Optional, Tuple
import numpy as np


class TreeModel:
    """
    Base class for tree models.

    Just ensures the presence of a tree member.
    """

    def __init__(self):
        self.tree: Optional[Tree] = None

    def show_tree(self) -> str:

        if self.tree is None:
            "Uninitialized tree model"

        return(tree_to_str(self.tree))


class AbstractTreeBuilder(ABC):
    """
    Abstract class for tree builders.

    Mostly used for type annotations.
    """

    @abstractmethod
    def _build(self):
        pass

    def prune(self): # Note: not abstract; no-op is a default pruning strategy
        pass


class TreeBuilder(AbstractTreeBuilder, TreeModel):
    """
    Greedy tree building strategy
    """

    def _build(self):

        root = self.new_node()
        self.tree = root.plant_tree()

        queue = self.new_queue()
        queue.push(root)

        while queue and not self.global_stop():

            node = queue.pop()
            node.split = self.construct_split(node)

            for branch in node.split.constraints:

                child = self.new_node(branch, node)
                node.add_child(child)

                if not self.local_stop(child):
                    queue.push(child)
    
    @abstractmethod
    def new_node(self, branch = None, parent = None):
        pass

    @abstractmethod
    def new_queue(self):
        pass

    @abstractmethod
    def construct_split(self, node):
        pass

    @abstractmethod
    def global_stop(self):
        pass

    @abstractmethod
    def local_stop(self, node):
        pass


class SplitTest(ABC):
    """
    Base abstract class for splits.

    A split is defined as a test with integer outcomes and a set of constraints each of which
    corresponds to an outcome of the test.
    """

    @abstractmethod
    def pick_branches(self, data_matrix):
        pass

    @property
    @abstractmethod
    def constraints(self):
        pass


class ClassificationTreeNode(TreeNode):
    """
    Mixin implementing classification tree node logic.

    Class must have a 'split' field deriving from SplitTest.
    """
    def __init__(self):
        super().__init__()
        self.split: SplitTest

    def node_proba(self, data_matrix):
        raise NotImplementedError

    def _predict_subtree_proba(self, data_matrix, idx, result):

        if self.is_leaf:
            result[idx,:] = self.node_proba(data_matrix[idx,])
        
        else:
            branches = self.split.pick_branches(data_matrix[idx,])
            for b in np.unique(branches):
                self[b]._predict_subtree_proba(data_matrix, idx[branches==b], result)

        return result


class TreeClassifierMixin(TreeModel):
    """
    The tree classifier mixin defines the logic of producing a classification using
    a decision tree.
    
    The tree (self.tree) should be used with a tree 'planted' from a subclass of
    ClassificationTreeNode, or something that monkeys it. That is where actual branching
    logic and prediction happens.
    The prediction functions also require a `target_classes` field to be defined.
    """

    def predict_proba(self, data_matrix):

        if self.tree is None:
            raise ValueError("Attempted to predict without an initialized tree.")

        n = data_matrix.shape[0]
        k = len(self.target_classes)

        result = np.empty((n, k), dtype=np.float)

        return self.tree.root._predict_subtree_proba(
            data_matrix,
            np.arange(n, dtype=np.intp),
            result)

    def predict(self, data_matrix):

        return self.target_classes[self.predict_proba(data_matrix).argmax(axis=1)]
