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


# Interface definition:
class NodeI(Protocol):

    data: np.ndarray
    y: np.ndarray
    local_constraint: Optional[Constraint] = None
    model: Optional[LocalEstimator] = None
    split: Optional[SplitTest] = None

# Implementations

# Supervised classification node adds nothing to the base interface
class Node(NodeI):
    pass

# Model translation node
class MTNode(NodeI):

    n_training: int
    coverage: float # Training set coverage
    constraints: Tuple[Constraint, ...]
    data_factory: DataFactoryLC

# Trepan node
@order_by('score')
class TrepanNode(MTNode):

    @cached_property
    def score(self):
        return -(self.coverage * (1 - self.fidelity))
    
    @cached_property
    def fidelity(self):
        hard_estimate = np.eye(self.y.shape[1])[self.model.estimate(self.data).argmax(axis=1),:]
        return sum((self.y * hard_estimate).mean(axis=0))



