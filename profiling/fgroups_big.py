
import cProfile, pstats
from generalizedtrees import learn, givens, predict, queues, stop, split, grow, leaves, generate, node
from time import perf_counter
import logging

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from collections import namedtuple

logger = logging.getLogger()

Dataset = namedtuple(
    'Dataset',
    ['x_train', 'x_test', 'y_train', 'y_test', 'feature_names', 'target_names'])

def mk_data():

    x_train = np.random.randint(2, size=(5000, 1000)).astype(bool)
    x_test = np.random.randint(2, size=(5000, 1000)).astype(bool)
    y_train = x_train[:, 0:99].sum(axis=1) > x_train[:, 200:299].sum(axis=1)
    y_test = x_test[:, 0:99].sum(axis=1) > x_test[:, 200:299].sum(axis=1)

    return Dataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(range(1000)),
        target_names=[True, False])

def get_rf_model(data):

    # Learn 'black-box' model
    rf = RandomForestClassifier()
    rf.fit(data.x_train, data.y_train)

    return rf

def profile_explanation(data, rf_model):
    x_train = data.x_train
    model = rf_model

    # Feature groups
    fg = [[x * 100 + i for i in range(100)] for x in range(10)]

    logger.info("Creating explainer object")

    explainer = learn.GreedyTreeLearner()
    explainer.builder.splitter = split.GroupSplitConstructorLC(search_mode='groups')
    explainer.builder.splitter.only_use_training_to_generate = True
    explainer.builder.splitter.only_use_training_to_score = False
    explainer.givens = givens.DataWithOracleGivensLC()
    explainer.predictor = predict.ClassifierLC()
    explainer.queue = queues.Heap
    explainer.local_stop = stop.LocalStopSaturation(training_only=True)
    explainer.global_stop = stop.GlobalStopTreeSizeLC(max_size=5)
    explainer.split_score = split.ProbabilityImpurityLC('entopy')
    explainer.split_generator = split.AxisAlignedSplitGeneratorLC()
    explainer.node_builder = grow.ModelTranslationNodeBuilderLC(
        leaf_model=leaves.ConstantEstimator,
        min_samples=1000,
        data_factory=generate.TrepanDataFactoryLC(alpha=0.05, max_attempts=3),
        node_type=node.TrepanNode,
    )

    logger.info("Fitting tree")

    with cProfile.Profile() as pr:
        explainer.fit(x_train, model, feature_groups = fg)
    
    ps = pstats.Stats(pr)
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()

def main():
    
    data = mk_data()

    profile_explanation(data, get_rf_model(data))

if __name__ == '__main__':
    main()