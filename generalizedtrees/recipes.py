# Recipes for known decision tree classifiers and model translators
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression

from generalizedtrees.node import MTNode, TrepanNode
from generalizedtrees.generate import SmearingDataFactoryLC, TrepanDataFactoryLC
from generalizedtrees.leaves import ConstantEstimator, SKProbaClassifier
from generalizedtrees.grow import ModelTranslationNodeBuilderLC, SupervisedNodeBuilderLC
from generalizedtrees.stop import GlobalStopTreeSizeLC, LocalStopDepthLC, LocalStopDisjunctionLC, LocalStopSaturation
from generalizedtrees.split import AxisAlignedSplitGeneratorLC, DiscreteInformationGainLC, IJCAI19LRGradientScoreLC, MofNSplitConstructorLC, ProbabilityImpurityLC
from generalizedtrees.queues import Heap, Queue, Stack
from generalizedtrees.predict import BinaryClassifierLC, ClassifierLC
from generalizedtrees.givens import DataWithOracleGivensLC, SupervisedDataGivensLC
from generalizedtrees.learn import GreedyTreeLearner


def binary_decision_tree_classifier(max_depth: int) -> GreedyTreeLearner:
    """
    Recipe for binary classification trees

    *Considering to deprecate this version
    Internal representation for y in this version is 1-d, representing the proportion of the 2nd
    (typically positive) class.
    """

    learner = GreedyTreeLearner()
    learner.givens = SupervisedDataGivensLC(binary_classification=True)
    learner.predictor = BinaryClassifierLC()
    learner.local_stop = LocalStopDepthLC(max_depth)
    learner.split_score = DiscreteInformationGainLC()
    learner.split_generator = AxisAlignedSplitGeneratorLC()
    learner.node_builder = SupervisedNodeBuilderLC(ConstantEstimator)
    learner.queue = Queue

    return learner


def decision_tree_classifier(max_depth: int = 20, impurity = 'entropy') -> GreedyTreeLearner:
    """
    Recipe for classification trees
    """

    learner = GreedyTreeLearner()
    learner.givens = SupervisedDataGivensLC()
    learner.predictor = ClassifierLC()
    learner.queue = Queue
    learner.local_stop = LocalStopDepthLC(max_depth)
    learner.split_score = ProbabilityImpurityLC(impurity)
    learner.split_generator = AxisAlignedSplitGeneratorLC()
    learner.node_builder = SupervisedNodeBuilderLC(ConstantEstimator)

    return learner


def trepan(
    m_of_n = False,
    max_tree_depth = None,
    max_tree_size = None,
    impurity = 'entropy',
    min_samples: int = 1000,
    dist_test_alpha = 0.05,
    max_attempts = 1000,
    rng = default_rng()
) -> GreedyTreeLearner:
    """
    Recipe for Trepan* (Craven and Shavlik 1995)

    *This version only implements axis-aligned splits (no m-of-n splits) and information gain as the split criterion.
    """

    learner = GreedyTreeLearner()
    if m_of_n:
        learner.builder.splitter = MofNSplitConstructorLC()
    learner.builder.splitter.only_use_training_to_generate = True
    learner.givens = DataWithOracleGivensLC()
    learner.predictor = ClassifierLC()
    learner.queue = Heap

    learner.local_stop = LocalStopSaturation(training_only=True)
    if max_tree_depth is not None:
        learner.local_stop = LocalStopDisjunctionLC(
            LocalStopSaturation(training_only=True),
            LocalStopDepthLC(max_depth=max_tree_depth))
    elif max_tree_size is not None:
        learner.global_stop = GlobalStopTreeSizeLC(max_tree_size)
    
    learner.split_score = ProbabilityImpurityLC(impurity)

    learner.split_generator = AxisAlignedSplitGeneratorLC()

    learner.node_builder = ModelTranslationNodeBuilderLC(
        leaf_model=ConstantEstimator,
        min_samples=min_samples,
        data_factory=TrepanDataFactoryLC(alpha=dist_test_alpha, max_attempts=max_attempts, rng=rng),
        node_type=TrepanNode)

    return learner


def trepan_logistic(
    max_tree_size: int = 20,
    min_samples: int = 1000,
    dist_test_alpha = 0.05,
    regularization_C = 0.1,
    criterion = 'entropy',
    max_attempts = 1000
) -> GreedyTreeLearner:
    """
    Recipe for Trepan with logistic regression at the leaves

    The source code for this recipe also serves as an illustration of how one can start with an existing recipe and
    modify select elements.
    """

    # Reuse simple trepan recipe
    learner = trepan(
        max_tree_size=max_tree_size,
        min_samples=min_samples,
        dist_test_alpha=dist_test_alpha,
        max_attempts=max_attempts)

    # Substitute a logistic learner as the leaf model
    learner.node_builder.new_model = lambda: SKProbaClassifier(
        LogisticRegression(
            penalty = 'l1',
            C = regularization_C,
            solver = 'saga',
            max_iter = 1000
        )
    )

    # Substitute alternate criterion if needed
    if criterion == 'gini':
        learner.split_score = ProbabilityImpurityLC('gini')
    elif criterion == 'ijcai2019':
        learner.split_score = IJCAI19LRGradientScoreLC()

    return learner


def born_again_tree(
    max_tree_size: int = 20,
    min_samples: int = 1000,
    impurity: str = 'entropy',
    p_alt: int = 0.5,
    max_attempts: int = 1000
) -> GreedyTreeLearner:
    """
    Recipe for Born Again Trees* ()

    *This version does not implement pruning and node cost calculation
    """

    learner = GreedyTreeLearner()
    learner.givens = DataWithOracleGivensLC()
    learner.predictor = ClassifierLC()
    learner.queue = Stack
    learner.global_stop = GlobalStopTreeSizeLC(max_tree_size)
    learner.split_score = ProbabilityImpurityLC(impurity)
    learner.split_generator = AxisAlignedSplitGeneratorLC()
    learner.node_builder = ModelTranslationNodeBuilderLC(
        leaf_model=ConstantEstimator,
        min_samples=min_samples,
        data_factory=SmearingDataFactoryLC(p_alt=p_alt, max_attempts=max_attempts),
        node_type=MTNode)

    return learner
