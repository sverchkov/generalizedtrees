# Fitting functions (for use in tree composition)
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

from generalizedtrees.base import AbstractTreeBuilder
from numpy import ndarray

def supervised_data_fit(tree_builder: AbstractTreeBuilder, data: ndarray, targets: ndarray, **kwargs):

    # TODO: Handle kwargs

    # Data checking can be inserted here

    tree_builder.data = data
    tree_builder.targets = targets

    tree_builder._build()
    tree_builder.prune()

    return tree_builder