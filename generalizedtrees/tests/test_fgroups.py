from generalizedtrees.features import FeatureSpec
import pytest

def test_fgroups_method(breast_cancer_data, breast_cancer_rf_model, caplog):

    from generalizedtrees import learn, givens, predict, queues, stop, split, grow, leaves, generate, node
    from time import perf_counter
    import logging

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    x_train = breast_cancer_data.x_train
    x_test = breast_cancer_data.x_test
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

    t1 = perf_counter()

    explainer.fit(x_train, model, feature_groups = fg)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    logger.info(f'Learned tree:\n{explainer.show_tree()}')

    # Make predictions
    logger.info("Running prediction")
    #trepan_predictions =
    explainer.predict(x_test)

    logger.info("Done")


def test_fgroups_method_bin_data(small_weights_data, small_weights_model, caplog):

    from generalizedtrees import learn, givens, predict, queues, stop, split, grow, leaves, generate, node
    from time import perf_counter
    import logging

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    x_train = small_weights_data.x_train
    x_test = small_weights_data.x_test
    model = small_weights_model

    # Feature groups
    fg = [[0,1,2,3], [3,4,5], [5,6,7,8,9], [12, 15]]

    logger.info("Creating explainer object")

    explainer = learn.GreedyTreeLearner()
    explainer.builder.splitter = split.GroupSplitConstructorLC(search_mode='groups_fast')
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

    t1 = perf_counter()

    explainer.fit(x_train, model, feature_groups = fg, feature_spec = (FeatureSpec.DISCRETE,) * 10)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    logger.info(f'Learned tree:\n{explainer.show_tree()}')

    # Make predictions
    logger.info("Running prediction")
    #trepan_predictions =
    explainer.predict(x_test)

    logger.info("Done")
