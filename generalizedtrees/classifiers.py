# Our implementation of standard decision tree classifiers
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

from dataclasses import field
from generalizedtrees.composing import compose_greedy_learner
from generalizedtrees.base import TreeBuilder
from generalizedtrees.fitters import supervised_data_fit
from generalizedtrees.splitters import construct_supervised_classifier_split, make_supervised_classifier_node
from generalizedtrees.queues import Stack
from generalizedtrees.stopping import tree_size_limit, perfect_classification

DecisionTreeClassifier = compose_greedy_learner(
    name="DecisionTreeClassifier",
    parameters=[
        ('max_tree_size', int, field(default=20))
    ],
    fitter=supervised_data_fit,
    construct_split=construct_supervised_classifier_split,
    new_node=make_supervised_classifier_node,
    queue=Stack,
    global_stop=tree_size_limit,
    local_stop=perfect_classification
)