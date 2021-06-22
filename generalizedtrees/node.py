# Tree learner node interface and implementations
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from functools import cached_property
from typing import Protocol, Optional, Tuple

import numpy as np

from generalizedtrees.constraints import Constraint
from generalizedtrees.generate import DataFactoryLC
from generalizedtrees.leaves import LocalEstimator
from generalizedtrees.split import SplitTest
from generalizedtrees.util import order_by


# Base definition:
class NodeBase:

    def __init__(self):
        self.data: np.ndarray
        self.y: np.ndarray
        self.local_constraint: Optional[Constraint] = None
        self.model: Optional[LocalEstimator] = None
        self.split: Optional[SplitTest] = None
        self.node_number: Optional[int] = None

# Supervised classification node adds nothing to the base interface
class Node(NodeBase):
    def __init__(self) -> None:
        super().__init__()

# Model translation node
class MTNode(NodeBase):

    def __init__(self) -> None:
        super().__init__()

        self.n_training: int
        self.coverage: float # Training set coverage
        self.constraints: Tuple[Constraint, ...]
        self.data_factory: DataFactoryLC

# Trepan node
@order_by('score')
class TrepanNode(MTNode):

    def __init__(self) -> None:
        super().__init__()

    @cached_property
    def score(self):
        return -(self.coverage * (1 - self.fidelity))
    
    @cached_property
    def fidelity(self):
        hard_estimate = np.eye(self.y.shape[1])[self.model.estimate(self.data).argmax(axis=1),:]
        return sum((self.y * hard_estimate).mean(axis=0))



