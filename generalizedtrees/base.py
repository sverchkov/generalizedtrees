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
from typing import Optional
import numpy as np


class TreeModel:

    def __init__(self):
        self.tree: Optional[Tree] = None

    def show_tree(self) -> str:

        if self.tree is None:
            "Uninitialized tree model"

        return(tree_to_str(self.tree))


class TreeBuilder(ABC, TreeModel):
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

            for branch in self.construct_split(node):

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


class TreeEstimatorNode(ABC, TreeNode):
    """
    Base abstract class for decision tree nodes.

    Formalizes the need to implement pick_branch and predict for use in tree
    estimators.
    """

    @abstractmethod
    def pick_branch(self, data_vector: np.ndarray) -> int:
        pass

    @abstractmethod
    def predict(self, data_vector: np.ndarray):
        pass


class TreeEstimatorMixin(TreeModel):
    """
    The tree estimator mixin defines the logic of producitg a prediction using
    a decision tree.
    
    It should be used with a tree 'planted' from a subclass of
    TreeEstimatorNode.
    The subclass of TreeEstimatorNode is where pick_branch (for internal nodes)
    and predict (for leaves) are defined.

    The decision logic is to follow branches (governed by pick_branch) and
    produce the decision at the leaves (governed by TreeEstimatorNode.predict).
    """

    def predict(self, data_matrix):
        return(np.apply_along_axis(self._predict1, 1, data_matrix))

    def _predict1(self, data_vector):

        if self.tree is None:
            raise ValueError("Attempted to predict without an initialized tree.")

        node = self.tree.root

        while not node.is_leaf:
            node = node[node.pick_branch(data_vector)]
        
        return(node.predict(data_vector))
