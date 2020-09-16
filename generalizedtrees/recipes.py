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
from generalizedtrees.fitters import supervised_data_fit, fit_with_data_and_oracle
from generalizedtrees.splitters import information_gain, information_gain_p, make_split_candidates, make_split_candidates_p
from generalizedtrees.queues import Stack, Heap
from generalizedtrees.stopping import never, node_depth, tree_size_limit
import generalizedtrees.node_building as nb
from generalizedtrees.data_generators import trepan_generator, smearing

DecisionTreeClassifier = greedy_classification_tree_learner(
    name="DecisionTreeClassifier",
    parameters=[
        ('max_depth', int, field(default=10))
    ],
    fitter=supervised_data_fit,
    node_building=nb.SupCferNodeBuilderMixin,
    split_candidate_generator=make_split_candidates,
    split_score=information_gain,
    queue=Stack,
    global_stop=never,
    local_stop=node_depth
)

# Version of Trepan that uses hard target estimation for split scoring
TrepanV1 = greedy_classification_tree_learner(
    name="Trepan",
    parameters=[
        ('use_m_of_n', bool, field(default=False)),
        ('max_tree_size', int, field(default=20)),
        ('min_samples', int, field(default=100)),
        ('dist_test_alpha', float, field(default=0.05))
    ],
    fitter=fit_with_data_and_oracle,
    node_building=nb.OGCferNodeBuilderMixin,
    split_candidate_generator=make_split_candidates,
    split_score=information_gain,
    data_generator=trepan_generator,
    queue=Heap,
    global_stop=tree_size_limit,
    local_stop=never,
    use_proba=False
)

# Version of Trepan that uses target probability estimates for split scoring
Trepan = greedy_classification_tree_learner(
    name="Trepan",
    parameters=[
        ('use_m_of_n', bool, field(default=False)),
        ('max_tree_size', int, field(default=20)),
        ('min_samples', int, field(default=100)),
        ('dist_test_alpha', float, field(default=0.05))
    ],
    fitter=fit_with_data_and_oracle,
    node_building=nb.OGCferNodeBuilderMixin,
    split_candidate_generator=make_split_candidates_p,
    split_score=information_gain_p,
    data_generator=trepan_generator,
    queue=Heap,
    global_stop=tree_size_limit,
    local_stop=never,
    use_proba=True
)

# Born again trees
BornAgain = greedy_classification_tree_learner(
    name="BornAgain",
    parameters=[
        ('max_tree_size', int, field(default=20)),
        ('min_samples', int, field(default=100))
    ],
    fitter=fit_with_data_and_oracle,
    node_building=nb.BATNodeBuilderMixin,
    split_candidate_generator=make_split_candidates_p,
    split_score=information_gain_p,
    data_generator=smearing,
    queue=Stack,
    global_stop=tree_size_limit,
    local_stop=never,
    use_proba=True
)

# Trepan with logistic regression leaves
TLL = greedy_classification_tree_learner(
        name="TLL",
    parameters=[
        ('node_cls', type, field(default=nb.TLLNode)),
        ('use_m_of_n', bool, field(default=False)),
        ('max_tree_size', int, field(default=20)),
        ('min_samples', int, field(default=100)),
        ('dist_test_alpha', float, field(default=0.05))
    ],
    fitter=fit_with_data_and_oracle,
    node_building=nb.OGCferNodeBuilderMixin,
    split_candidate_generator=make_split_candidates_p,
    split_score=information_gain_p,
    data_generator=trepan_generator,
    queue=Heap,
    global_stop=tree_size_limit,
    local_stop=never,
    use_proba=True
)