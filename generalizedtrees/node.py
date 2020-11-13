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

from abc import abstractmethod
from typing import Iterable, Protocol, Optional

import numpy as np

from generalizedtrees.leaves import LocalEstimator
from generalizedtrees.constraints import Constraint
from generalizedtrees.base import SplitTest


# Interface definition:
class Node(Protocol):

    data: np.ndarray
    y: np.ndarray
    local_constraint: Optional[Constraint] = None
    model: Optional[LocalEstimator] = None
    split: Optional[SplitTest] = None

    @abstractmethod
    def generate_children(self) -> Iterable['Node']:
        raise NotImplementedError


