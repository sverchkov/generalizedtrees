# Tests for trepan
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

import pytest
from numpy.testing import assert_allclose

def test_trepan_numpy(breast_cancer_data, breast_cancer_rf_model, caplog):

    from generalizedtrees.recipes import trepan
    from generalizedtrees.features import FeatureSpec
    from generalizedtrees.split import ProbabilityImpurityLC
    from time import perf_counter
    import logging

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    x_train = breast_cancer_data.x_train
    x_test = breast_cancer_data.x_test
    model = breast_cancer_rf_model

    # Learn explanation
    d = x_train.shape[1]

    t1 = perf_counter()

    logger.info("Creating class instance")
    explanation = trepan(max_tree_size = 5, max_attempts=3)
    explanation.split_score = ProbabilityImpurityLC('gini')
    explanation.builder.splitter.infimum_score_to_split = -10

    logger.info("Fitting tree")
    explanation.fit(x_train, model.predict_proba, feature_spec = (FeatureSpec.CONTINUOUS,)*d)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    logger.info(f'Learned tree:\n{explanation.show_tree()}')

    # Make predictions
    logger.info("Running prediction")
    #trepan_predictions =
    explanation.predict(x_test)

    logger.info("Done")


@pytest.mark.skip(reason="Very slow")
def test_real_trepan(bc_lowD_data, bc_lowD_rf_model, caplog):

    from generalizedtrees.recipes import trepan
    from generalizedtrees.split import ProbabilityImpurityLC
    from time import perf_counter
    import logging

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    x_train = bc_lowD_data.x_train
    x_test = bc_lowD_data.x_test
    model = bc_lowD_rf_model

    # Learn explanation
    d = x_train.shape[1]

    t1 = perf_counter()

    logger.info("Creating class instance")
    explanation = trepan(max_tree_size = 6, max_attempts=3, m_of_n = True)
    #explanation.split_score = ProbabilityImpurityLC('gini')
    #explanation.builder.splitter.infimum_score_to_split = -10

    logger.info("Fitting tree")
    explanation.fit(x_train, model.predict_proba)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    logger.info(f'Learned tree:\n{explanation.show_tree()}')

    # Make predictions
    logger.info("Running prediction")
    #trepan_predictions =
    explanation.predict(x_test)

    logger.info("Done")


def test_trepan_pandas(breast_cancer_data_pandas, breast_cancer_rf_model, caplog):

    from generalizedtrees.recipes import trepan
    import pandas as pd
    from time import perf_counter
    import logging

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    x_train = breast_cancer_data_pandas.x_train
    x_test = breast_cancer_data_pandas.x_test
    model = breast_cancer_rf_model
    target_names = breast_cancer_data_pandas.target_names

    # Verify output shape of model
    logger.debug(f'Model probability prediction:\n{model.predict_proba(x_test)}')   

    # Learn explanation
    t1 = perf_counter()

    logger.info('Creating class instance')
    trepan = trepan(max_tree_size = 5, max_attempts=3)

    logger.info('Fitting tree')
    oracle = lambda x: model.predict_proba(x)
    trepan.fit(x_train, oracle)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    logger.info(f'Learned tree:\n{trepan.show_tree()}')

    # Make predictions
    logger.info('Running prediction')
    trepan_predictions = trepan.predict(x_test)

    logger.info(f'Predictions: {list(trepan_predictions)}')

    logger.info("Done")

