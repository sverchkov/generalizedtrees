# Tree growing algorithms
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from abc import abstractmethod
from generalizedtrees.generate import DataFactoryLC

from generalizedtrees.givens import DataWithOracleGivensLC, GivensLC, SupervisedDataGivensLC
from generalizedtrees.leaves import LocalEstimator

import numpy as np
from generalizedtrees.node import MTNode, Node, NodeI
from typing import Callable, Generic, Iterable, Optional, Protocol, Type, TypeVar

from generalizedtrees.queues import CanPushPop
from generalizedtrees.split import SplitConstructorLC, DefaultSplitConstructorLC
from generalizedtrees.stop import GlobalStopLC, LocalStopLC, NeverStopLC
from generalizedtrees.tree import Tree


###########################
# Node builder components #
###########################

N = TypeVar('N', bound=NodeI)

# Interface definition
class NodeBuilderLC(Protocol, Generic[N]):

    node_type: Type[N]
    new_model: Callable[[], LocalEstimator]

    @abstractmethod
    def initialize(self, givens: GivensLC) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_root(self) -> N:
        raise NotImplementedError

    @abstractmethod
    def generate_children(self, node: N) -> Iterable[N]:
        raise NotImplementedError


# Implementations

# For supervised learning
class SupervisedNodeBuilderLC(NodeBuilderLC[Node]):

    data: np.ndarray
    y: np.ndarray

    def __init__(self, leaf_model: Callable[[], LocalEstimator]) -> None:
        self.node_type = Node
        self.new_model = leaf_model

    def initialize(self, givens: GivensLC) -> None:
        assert(isinstance(givens, SupervisedDataGivensLC))
        self.data = givens.data_matrix
        self.y = givens.target_matrix
    
    def create_root(self) -> Node:
        node = self.node_type()
        node.data = self.data
        node.y = self.y
        node.model = self.new_model().fit(node.data, node.y)
        return node
    
    def generate_children(self, node: Node) -> Iterable[Node]:
        if node.split is not None:
            branches = node.split.pick_branches(node.data)
            # TODO: Check that data goes into multiple branches
            # Or should it be that some data goes to each branch?
            for b, c in enumerate(node.split.constraints):
                idx = branches == b
                # Possible patch: if idx is empty, copy model from parent
                child = self.node_type()
                child.data = node.data[idx, :]
                child.y = node.y[idx, :]
                child.model = self.new_model().fit(child.data, child.y)
                child.local_constraint = c
                yield child
              

# For model translation
class ModelTranslationNodeBuilderLC(NodeBuilderLC[MTNode]):

    min_samples: int

    training_data: np.ndarray
    training_y: Optional[np.ndarray]
    oracle: Callable
    data_factory: DataFactoryLC


    def __init__(
        self,
        leaf_model: Callable[[], LocalEstimator],
        min_samples: int,
        data_factory: DataFactoryLC,
        node_type: Type[MTNode] = MTNode
    ) -> None:
        self.node_type = node_type
        self.new_model = leaf_model
        self.min_samples = min_samples
        self.data_factory = data_factory
    

    def initialize(self, givens: GivensLC) -> None:
        assert(isinstance(givens, DataWithOracleGivensLC))
        self.training_data = givens.data_matrix
        self.training_y = givens.target_matrix
        self.oracle = givens.oracle
        self.data_factory.feature_spec = givens.feature_spec
    
    def create_root(self) -> MTNode:

        node = self.node_type()

        node.n_training = self.training_data.shape[0]

        node.data_factory = self.data_factory.refit(self.training_data)

        gen_data = node.data_factory.generate(self.min_samples - node.n_training)

        node.data = np.row_stack((self.training_data, gen_data))

        node.y = self.oracle(node.data)

        node.model = self.new_model().fit(node.data, node.y)

        node.constraints = ()

        node.coverage = 1

        return node

    def generate_children(self, node: MTNode) -> Iterable[MTNode]:

        if node.split is not None:
            branches = node.split.pick_branches(node.data)
            for b, c in enumerate(node.split.constraints):
                idx = branches == b

                child = self.node_type()

                child.n_training = idx[0:node.n_training].sum()
                pregen_data = node.data[idx, :]

                child.data_factory = node.data_factory.refit(pregen_data[0:child.n_training, :])

                gen_data = child.data_factory.generate(self.min_samples - pregen_data.shape[0])
                if gen_data.shape[0] < 1:
                    gen_y = np.empty((0, node.y.shape[1]))
                else:
                    gen_y = self.oracle(gen_data)

                child.data = np.row_stack((pregen_data, gen_data))
                child.y = np.row_stack((node.y[idx, :], gen_y))

                child.model = self.new_model().fit(child.data, child.y)

                child.local_constraint = c
                child.constraints = node.constraints + (c,)

                if node.n_training <= 0:
                    child.coverage = 0
                else:
                    child.coverage = node.coverage * sum(idx[0:node.n_training]) / node.n_training

                yield child


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
    global_stop: GlobalStopLC = NeverStopLC()
    local_stop: LocalStopLC = NeverStopLC()

    def __init__(self):
        self.splitter = DefaultSplitConstructorLC()
    
    def initialize(self, givens: GivensLC) -> None:
        self.node_builder.initialize(givens)
        self.splitter.initialize(givens)

    def build_tree(self) -> Tree:

        root = self.node_builder.create_root()
        tree = Tree([root])

        queue = self.new_queue()
        queue.push((root, 'root'))

        while queue and not self.global_stop.check(tree):

            node, ptr = queue.pop()

            if not self.local_stop.check(tree.node(ptr)):

                node.split = self.splitter.construct_split(node)

                if node.split:
                    for child in self.node_builder.generate_children(node):
                        child_ptr = tree.add_node(child, parent_key=ptr)
                        queue.push((child, child_ptr))
        
        return tree
    
    def prune_tree(self, tree: Tree):
        return tree
