# Tree learner node base class and node factories
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020-2021, Yuriy Sverchkov

from abc import abstractmethod
from functools import cached_property
from typing import Callable, Generic, Iterable, Optional, Protocol, Tuple, Type, TypeVar

import numpy as np

from generalizedtrees.constraints import Constraint
from generalizedtrees.generate import DataFactoryLC
from generalizedtrees.givens import DataWithOracleGivensLC, GivensLC, SupervisedDataGivensLC
from generalizedtrees.leaves import LocalEstimator
from generalizedtrees.scores import soft_hard_product_loss
from generalizedtrees.split import SplitTest
from generalizedtrees.util import order_by


##################################
# Base and interface definitions #
##################################

# Base definition for node:
class NodeBase:

    def __init__(self):
        # "Declares" member attributes that are used regardless of specific tree implementation 
        self.local_constraint: Optional[Constraint] = None
        self.model: Optional[LocalEstimator] = None
        self.split: Optional[SplitTest] = None
        self.node_number: Optional[int] = None

N = TypeVar('N', bound=NodeBase)

# Node builder interface definition
class NodeBuilderLC(Protocol):

    @abstractmethod
    def initialize(self, givens: GivensLC) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_root(self) -> Tuple[N, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def generate_children(self, node: N) -> Iterable[Tuple[N, np.ndarray, np.ndarray]]:
        raise NotImplementedError

###################
# Implementations #
###################

# For supervised learning
#########################

# Node class
class Node(NodeBase):
    def __init__(self):
        super().__init__()

# Node builder
class SupervisedNodeBuilderLC(NodeBuilderLC):

    def __init__(self, leaf_model: Callable[[], LocalEstimator]) -> None:
        self.data: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.new_model = leaf_model

    def initialize(self, givens: GivensLC) -> None:
        assert(isinstance(givens, SupervisedDataGivensLC))
        self.data = givens.data_matrix
        self.y = givens.target_matrix
    
    def create_root(self) -> Tuple[Node, np.ndarray, np.ndarray]:
        node = Node()
        node.model = self.new_model().fit(self.data, self.y)
        return node, self.data, self.y
    
    def generate_children(self, node: Node, data, y) -> Iterable[Tuple[Node, np.ndarray, np.ndarray]]:
        if node.split is not None:
            branches = node.split.pick_branches(data)
            # TODO: Check that data goes into multiple branches
            # Or should it be that some data goes to each branch?
            for b, c in enumerate(node.split.constraints):
                idx = branches == b
                # Possible patch: if idx is empty, copy model from parent (???)
                child = Node()
                c_data = data[idx]
                c_y = y[idx]
                child.model = self.new_model().fit(c_data, c_y)
                child.local_constraint = c
                yield child, c_data, c_y


# For model translation
#######################

# Model translation node
class MTNode(NodeBase):

    def __init__(self, n_training: int) -> None:
        super().__init__()

        self.n_training = n_training
        self.coverage: Optional[float] = None # Training set coverage
        self.constraints: Optional[Tuple[Constraint, ...]] = None
        self.data_factory: Optional[DataFactoryLC] = None

    def fit(self, data: np.ndarray, y: np.ndarray) -> 'MTNode':

        self.model.fit(data, y)

        return self

# Trepan node
@order_by('score')
class TrepanNode(MTNode):

    def __init__(
        self,
        n_training: int,
        fidelity_loss: Callable[[np.ndarray, np.ndarray], float] = soft_hard_product_loss
        ) -> None:

        super().__init__(n_training)

        self.fidelity_loss_fn = fidelity_loss
        self.fidelity: Optional[float] = None
        self.score: Optional[float] = None

    def fit(self, data: np.ndarray, y: np.ndarray) -> 'TrepanNode':

        super(TrepanNode, self).fit(data, y)

        fidelity_loss = self.fidelity_loss_fn(y, self.model.estimate(data))

        self.fidelity = 1 - fidelity_loss
        self.score = -(self.coverage * fidelity_loss)

        return self

# For model translation
class ModelTranslationNodeBuilderLC(NodeBuilderLC):

    def __init__(
        self,
        leaf_model: Callable[[], LocalEstimator],
        min_samples: int,
        data_factory: DataFactoryLC,
        node_type: Type[MTNode] = MTNode
    ) -> None:

        self.node_type: Type[MTNode] = node_type
        self.new_model: Callable[[], LocalEstimator] = leaf_model
        self.min_samples: int = min_samples
        self.data_factory: DataFactoryLC = data_factory
    
        self.training_data: Optional[np.ndarray] = None
        self.training_y: Optional[np.ndarray] = None
        self.oracle: Optional[Callable] = None

    def initialize(self, givens: GivensLC) -> None:
        assert(isinstance(givens, DataWithOracleGivensLC))
        self.training_data = givens.data_matrix
        self.training_y = givens.target_matrix
        self.oracle = givens.oracle
        self.data_factory.feature_spec = givens.feature_spec
    
    def create_root(self) -> MTNode:

        n_training = self.training_data.shape[0]

        node = self.node_type(n_training)

        node.data_factory = self.data_factory.refit(self.training_data)

        gen_data = node.data_factory.generate(self.min_samples - n_training)

        data = np.row_stack((self.training_data, gen_data))
        y = self.oracle(data)

        node.model = self.new_model()
        node.constraints = ()
        node.coverage = 1

        node.fit(data, y)

        return node, data, y

    def generate_children(self, node: MTNode, data: np.ndarray, y: np.ndarray) -> Iterable[MTNode]:

        if node.split is not None:
            branches = node.split.pick_branches(data)
            for b, c in enumerate(node.split.constraints):
                idx = branches == b

                child = self.node_type(sum(idx[:node.n_training]))

                pregen_data = data[idx, :]
                pregen_y = y[idx, :]

                if (pregen_data.shape[0] < self.min_samples):
                    child.data_factory = node.data_factory.refit(pregen_data[:child.n_training, :])

                    gen_data = child.data_factory.generate(self.min_samples - pregen_data.shape[0])
                    gen_y = self.oracle(gen_data)

                else:
                    child.data_factory = node.data_factory
                    gen_data = np.empty((0, pregen_data.shape[1]))
                    gen_y = np.empty((0, y.shape[1]))

                child_data = np.row_stack((pregen_data, gen_data))
                child_y = np.row_stack((pregen_y, gen_y))

                child.model = self.new_model()

                child.local_constraint = c
                child.constraints = node.constraints + (c,)

                if node.n_training <= 0:
                    child.coverage = 0
                else:
                    child.coverage = node.coverage * sum(idx[:node.n_training]) / node.n_training

                child.fit(child_data, child_y)

                yield child, child_data, child_y


