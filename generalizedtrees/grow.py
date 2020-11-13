# Tree growing algorithms
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

from abc import abstractmethod
from generalizedtrees.givens import GivensLC, SupervisedDataGivensLC
from generalizedtrees.leaves import LocalEstimator

import numpy as np
from generalizedtrees.node import Node
from typing import Callable, Iterable, Protocol

from generalizedtrees.base import SplitTest
from generalizedtrees.queues import CanPushPop
from generalizedtrees.split import SplitConstructorLC
from generalizedtrees.stop import GlobalStopLC, LocalStopLC
from generalizedtrees.tree import Tree


###########################
# Node builder components #
###########################

# Interface definition
class NodeBuilderLC(Protocol):

    @abstractmethod
    def initialize(self, givens: GivensLC) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_root(self) -> Node:
        raise NotImplementedError

    @abstractmethod
    def generate_children(self, node: Node) -> Iterable[Node]:
        raise NotImplementedError

# Implementations
class SupervisedNodeBuilderLC(NodeBuilderLC):

    new_model: Callable[[], LocalEstimator]
    data: np.ndarray
    y: np.ndarray

    def initialize(self, givens: GivensLC) -> None:
        assert(isinstance(givens, SupervisedDataGivensLC))
        self.data = givens.data_matrix
        self.y = givens.target_matrix


class ModelTranslationNodeBuilderLC(NodeBuilderLC):

    training_data: np.ndarray
    oracle: Callable
    data_generator: Callable


###########################
# Tree builder components #
###########################

# Future: define interface protocol if we will ever have other builder strategies

class GreedyBuilderLC:
    """
    Greedy tree building strategy
    """

    new_queue: Callable[..., CanPushPop]
    node_builder: NodeBuilderLC
    splitter: SplitConstructorLC
    global_stop: GlobalStopLC
    local_stop: LocalStopLC

    def build_tree(self):

        root = self.node_builder.create_root()
        tree = Tree([root])

        queue = self.new_queue()
        queue.push((root, 'root'))

        while queue and not self.global_stop.check(tree):

            node, ptr = queue.pop()

            if not self.local_stop.check(tree.node(ptr)):

                node.split = self.splitter.construct_split(node)

                if node.split is not None:
                    for child in self.node_builder.generate_children(node):
                        child_ptr = tree.add_node(child, parent_key=ptr)
                        queue.push((child, child_ptr))
        
        return tree
    
    def prune_tree(self, tree):
        return tree
