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
from generalizedtrees.composing import greedy_classification_tree_learner
from generalizedtrees.base import TreeBuilder
from generalizedtrees.fitters import supervised_data_fit, fit_with_data_and_oracle
from generalizedtrees.splitters import information_gain, make_split_candidates
from generalizedtrees.queues import Stack, Heap
from generalizedtrees.stopping import never, node_depth, tree_size_limit
from generalizedtrees.node_building import SupCferNodeBuilderMixin, OGCferNodeBuilderMixin
from generalizedtrees.data_generators import trepan_generator

DecisionTreeClassifier = greedy_classification_tree_learner(
    name="DecisionTreeClassifier",
    parameters=[
        ('max_depth', int, field(default=10))
    ],
    fitter=supervised_data_fit,
    node_building=SupCferNodeBuilderMixin,
    split_candidate_generator=make_split_candidates,
    split_score=information_gain,
    queue=Stack,
    global_stop=never,
    local_stop=node_depth
)

Trepan = greedy_classification_tree_learner(
    name="Trepan",
    parameters=[
        ('max_tree_size', int, field(default=20)),
        ('min_samples', int, field(default=100)),
        ('dist_test_alpha', float, field(default=0.05))
    ],
    fitter=fit_with_data_and_oracle,
    node_building=OGCferNodeBuilderMixin,
    split_candidate_generator=make_split_candidates,
    split_score=information_gain,
    data_generator=trepan_generator,
    queue=Heap,
    global_stop=tree_size_limit,
    local_stop=never
)