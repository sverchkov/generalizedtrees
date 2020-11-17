# Tests for born-again trees
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

def test_tll_entropy(breast_cancer_data_pandas, breast_cancer_rf_model, caplog):

    from generalizedtrees.recipes import trepan_logistic
    from generalizedtrees.vis import explanation_to_JSON
    from generalizedtrees.vis.vis import explanation_to_simplified
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
    explain = trepan_logistic(max_attempts=3)

    logger.info('Fitting tree')
    oracle = model.predict_proba
    explain.fit(x_train, oracle)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    # Test simple printing
    logger.info(f'Learned tree:\n{explain.show_tree()}')

    # Test JSON
    logger.info(f'Simplified: {explanation_to_simplified(explain)}')
    logger.info(f'JSON: {explanation_to_JSON(explain)}')

    # Make predictions
    logger.info('Running prediction')
    explainer_predictions = explain.predict(x_test)

    logger.info(f'Predictions: {list(explainer_predictions)}')

    logger.info("Done")


def test_tll_ijcai2019(breast_cancer_data_pandas, breast_cancer_rf_model, caplog):

    from generalizedtrees.recipes import trepan_logistic
    from generalizedtrees.vis import explanation_to_JSON
    from generalizedtrees.vis.vis import explanation_to_simplified
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
    explain = trepan_logistic(criterion='ijcai2019', max_attempts=3)

    logger.info('Fitting tree')
    oracle = model.predict_proba
    explain.fit(x_train, oracle)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    # Test simple printing
    logger.info(f'Learned tree:\n{explain.show_tree()}')

    # Test JSON
    logger.info(f'Simplified: {explanation_to_simplified(explain)}')
    logger.info(f'JSON: {explanation_to_JSON(explain)}')

    # Make predictions
    logger.info('Running prediction')
    explainer_predictions = explain.predict(x_test)

    logger.info(f'Predictions: {list(explainer_predictions)}')

    logger.info("Done")