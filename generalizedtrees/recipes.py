# Recipes for known decision tree classifiers and model translators
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


from generalizedtrees.node import TrepanNode
from generalizedtrees.generate import TrepanDataFactoryLC
from generalizedtrees.leaves import ConstantEstimator
from generalizedtrees.grow import ModelTranslationNodeBuilderLC, SupervisedNodeBuilderLC
from generalizedtrees.stop import GlobalStopTreeSizeLC, LocalStopDepthLC
from generalizedtrees.split import AxisAlignedSplitGeneratorLC, DiscreteInformationGainLC, ProbabilityImpurityLC
from generalizedtrees.queues import Heap, Queue
from generalizedtrees.predict import BinaryClassifierLC, ClassifierLC
from generalizedtrees.givens import DataWithOracleGivensLC, SupervisedDataGivensLC
from generalizedtrees.learn import GreedyTreeLearner


def binary_decision_tree_classifier(max_depth: int) -> GreedyTreeLearner:

    learner = GreedyTreeLearner()
    learner.givens = SupervisedDataGivensLC(binary_classification=True)
    learner.predictor = BinaryClassifierLC()
    learner.local_stop = LocalStopDepthLC(max_depth)
    learner.split_score = DiscreteInformationGainLC()
    learner.split_generator = AxisAlignedSplitGeneratorLC()
    learner.node_builder = SupervisedNodeBuilderLC(ConstantEstimator)
    learner.queue = Queue

    return learner


def decision_tree_classifier(max_depth: int, impurity = 'entropy') -> GreedyTreeLearner:

    learner = GreedyTreeLearner()
    learner.givens = SupervisedDataGivensLC()
    learner.predictor = ClassifierLC()
    learner.queue = Queue
    learner.local_stop = LocalStopDepthLC(max_depth)
    learner.split_score = ProbabilityImpurityLC(impurity)
    learner.split_generator = AxisAlignedSplitGeneratorLC()
    learner.node_builder = SupervisedNodeBuilderLC(ConstantEstimator)

    return learner


def trepan(max_tree_size: int = 20, min_samples: int = 1000, dist_test_alpha = 0.05, max_attempts = 1000):

    learner = GreedyTreeLearner()
    learner.givens = DataWithOracleGivensLC()
    learner.predictor = ClassifierLC()
    learner.queue = Heap
    learner.global_stop = GlobalStopTreeSizeLC(max_tree_size)
    learner.split_score = ProbabilityImpurityLC('entropy')
    learner.split_generator = AxisAlignedSplitGeneratorLC()
    learner.node_builder = ModelTranslationNodeBuilderLC(
        leaf_model=ConstantEstimator,
        min_samples=min_samples,
        data_factory=TrepanDataFactoryLC(alpha=dist_test_alpha, max_attempts=max_attempts),
        node_type=TrepanNode)

    return learner

# Version of Trepan that uses hard target estimation for split scoring
# TrepanV1 = greedy_classification_tree_learner(
#     name="Trepan",
#     parameters=[
#         ('use_m_of_n', bool, field(default=False)),
#         ('max_tree_size', int, field(default=20)),
#         ('min_samples', int, field(default=100)),
#         ('dist_test_alpha', float, field(default=0.05))
#     ],
#     fitter=fit_with_data_and_oracle,
#     node_building=nb.OGCferNodeBuilderMixin,
#     split_candidate_generator=make_split_candidates,
#     split_score=information_gain,
#     data_generator=trepan_generator,
#     queue=Heap,
#     global_stop=tree_size_limit,
#     local_stop=never,
#     use_proba=False
# )

# # Version of Trepan that uses target probability estimates for split scoring
# Trepan = greedy_classification_tree_learner(
#     name="Trepan",
#     parameters=[
#         ('use_m_of_n', bool, field(default=False)),
#         ('max_tree_size', int, field(default=20)),
#         ('min_samples', int, field(default=100)),
#         ('dist_test_alpha', float, field(default=0.05))
#     ],
#     fitter=fit_with_data_and_oracle,
#     node_building=nb.OGCferNodeBuilderMixin,
#     split_candidate_generator=make_split_candidates_p,
#     split_score=information_gain_p,
#     data_generator=trepan_generator_n,
#     queue=Heap,
#     global_stop=tree_size_limit,
#     local_stop=never,
#     use_proba=True
# )

# # Born again trees
# BornAgain = greedy_classification_tree_learner(
#     name="BornAgain",
#     parameters=[
#         ('max_tree_size', int, field(default=20)),
#         ('min_samples', int, field(default=100))
#     ],
#     fitter=fit_with_data_and_oracle,
#     node_building=nb.BATNodeBuilderMixin,
#     split_candidate_generator=make_split_candidates_p,
#     split_score=information_gain_p,
#     data_generator=smearing,
#     queue=Stack,
#     global_stop=tree_size_limit,
#     local_stop=never,
#     use_proba=True
# )

# # Trepan with logistic regression leaves
# TLL = greedy_classification_tree_learner(
#         name="TLL",
#     parameters=[
#         ('node_cls', type, field(default=nb.TLLNode)),
#         ('use_m_of_n', bool, field(default=False)),
#         ('max_tree_size', int, field(default=20)),
#         ('min_samples', int, field(default=100)),
#         ('dist_test_alpha', float, field(default=0.05))
#     ],
#     fitter=fit_with_data_and_oracle,
#     node_building=nb.OGCferNodeBuilderMixin,
#     split_candidate_generator=make_split_candidates_p,
#     split_score=information_gain_p,
#     data_generator=trepan_generator_n,
#     queue=Heap,
#     global_stop=tree_size_limit,
#     local_stop=never,
#     use_proba=True
# )

# # Trepan with logistic regression leaves
# TLL2 = greedy_classification_tree_learner(
#         name="TLL",
#     parameters=[
#         ('node_cls', type, field(default=nb.TLLNode)),
#         ('use_m_of_n', bool, field(default=False)),
#         ('max_tree_size', int, field(default=20)),
#         ('min_samples', int, field(default=100)),
#         ('dist_test_alpha', float, field(default=0.05))
#     ],
#     fitter=fit_with_data_and_oracle,
#     node_building=nb.OGCferNodeBuilderMixin,
#     split_candidate_generator=make_split_candidates_p,
#     split_score=ijcai19_lr_gradient_slow,
#     data_generator=trepan_generator_n,
#     queue=Heap,
#     global_stop=tree_size_limit,
#     local_stop=never,
#     use_proba=True,
#     splits_composer=compose_split_constructor_mk2
# )

# # Trepan with logistic regression leaves and a 1-d estimator
# TLL3 = greedy_binary_classification_tree_learner(
#         name="TLL",
#     parameters=[
#         ('node_cls', type, field(default=nb.BTLLNode)),
#         ('use_m_of_n', bool, field(default=False)),
#         ('max_tree_size', int, field(default=20)),
#         ('min_samples', int, field(default=100)),
#         ('dist_test_alpha', float, field(default=0.05))
#     ],
#     fitter=fit_with_data_and_oracle,
#     node_building=nb.OGCferNodeBuilderMixin,
#     split_candidate_generator=make_split_candidates_p,
#     split_score=ijcai19_lr_gradient_slow,
#     data_generator=trepan_generator_n,
#     queue=Heap,
#     global_stop=tree_size_limit,
#     local_stop=never,
#     use_proba=True,
#     splits_composer=compose_split_constructor_mk2
# )