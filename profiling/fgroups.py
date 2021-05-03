
import cProfile, pstats
from generalizedtrees import learn, givens, predict, queues, stop, split, grow, leaves, generate, node
from time import perf_counter
import logging

from numpy.random import seed
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from collections import namedtuple

logger = logging.getLogger()

def get_breast_cancer_data():

    # Load data
    bc = load_breast_cancer()

    x_train, x_test, y_train, y_test = train_test_split(bc['data'], bc['target'])

    Dataset = namedtuple(
        'Dataset',
        ['x_train', 'x_test', 'y_train', 'y_test', 'feature_names', 'target_names'])

    return Dataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=bc['feature_names'],
        target_names=bc['target_names'])

def get_breast_cancer_rf_model(breast_cancer_data):

    # Learn 'black-box' model
    rf = RandomForestClassifier()
    rf.fit(breast_cancer_data.x_train, breast_cancer_data.y_train)

    return rf

def profile_explanation(breast_cancer_data, breast_cancer_rf_model):
    x_train = breast_cancer_data.x_train
    model = breast_cancer_rf_model

    # Feature groups
    fg = [[x, x+10, x+20] for x in range(10)]

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
    
    bc = get_breast_cancer_data()

    profile_explanation(bc, get_breast_cancer_rf_model(bc))

if __name__ == '__main__':
    main()