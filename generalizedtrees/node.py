# Tree learner node interface and implementations
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
    coverage: float
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
        return sum((self.y * self.model.estimate(self.data)).mean(axis=0))



