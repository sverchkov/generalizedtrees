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

from typing import Protocol, Optional, Tuple

import numpy as np

from generalizedtrees.base import SplitTest
from generalizedtrees.constraints import Constraint
from generalizedtrees.generate import DataFactoryLC
from generalizedtrees.leaves import LocalEstimator


# Interface definition:
class Node(Protocol):

    data: np.ndarray
    y: np.ndarray
    local_constraint: Optional[Constraint] = None
    model: Optional[LocalEstimator] = None
    split: Optional[SplitTest] = None

# Implementations

# Model translation node
class MTNode(Node):

    n_training: int
    constraints: Tuple[Constraint, ...]
    data_factory: DataFactoryLC



