# Tests for regular classification trees
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

import pytest
from numpy.testing import assert_allclose

def test_dtc(breast_cancer_data, caplog):

    import logging
    from generalizedtrees.recipes import binary_decision_tree_classifier
    from generalizedtrees.features import FeatureSpec

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = binary_decision_tree_classifier(max_depth = 5)

    logger.info("Fitting tree")
    d = breast_cancer_data.x_train.shape[1]

    dtc.fit(breast_cancer_data.x_train, breast_cancer_data.y_train, feature_spec=(FeatureSpec.CONTINUOUS,)*d)

    logger.info(f'Learned tree:\n{dtc.show_tree()}')

    logger.info("Running prediction")
    dtc.predict(breast_cancer_data.x_test)

    logger.info("Done")


@pytest.mark.skip(reason="Don't know all the details to sklearn's implementation")
def test_dtc_prediction(breast_cancer_data, caplog):
    import logging
    from generalizedtrees.recipes import decision_tree_classifier
    from generalizedtrees.features import FeatureSpec
    from sklearn.tree import DecisionTreeClassifier as SKDTC
    from sklearn.tree import export_text

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = decision_tree_classifier(max_depth = 5, impurity='gini')

    logger.info("Fitting tree")
    d = breast_cancer_data.x_train.shape[1]

    dtc.fit(breast_cancer_data.x_train, breast_cancer_data.y_train, feature_spec=(FeatureSpec.CONTINUOUS,)*d)

    logger.info(f'Learned tree:\n{dtc.show_tree()}')

    logger.info("Running prediction")
    my_pr = dtc.predict(breast_cancer_data.x_test)

    logger.info("Running Scikit-Learn's DT")
    sk_dtc = SKDTC(max_depth=5)
    sk_dtc.fit(breast_cancer_data.x_train, breast_cancer_data.y_train)

    logger.info(f'SKLearn learned tree:\n{export_text(sk_dtc)}')

    logger.info('Running SKLearn prediction')
    sk_pr = sk_dtc.predict(breast_cancer_data.x_test)

    logger.info("Comparing")
    assert_allclose(my_pr, sk_pr)

    logger.info("Done")


def test_dtc_pandas(breast_cancer_data_pandas, caplog):
    import logging
    from generalizedtrees.recipes import binary_decision_tree_classifier

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = binary_decision_tree_classifier(max_depth = 5)

    logger.info("Fitting tree")

    dtc.fit(breast_cancer_data_pandas.x_train, breast_cancer_data_pandas.y_train)

    logger.info(f'Learned tree:\n{dtc.show_tree()}')

    logger.info("Running prediction")
    dtc.predict(breast_cancer_data_pandas.x_test)

    logger.info("Done")


def test_dtc_bin_vs_multi(breast_cancer_data_pandas, caplog):
    import logging
    from generalizedtrees.recipes import binary_decision_tree_classifier, decision_tree_classifier

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating binary version")
    dtc_bin = binary_decision_tree_classifier(max_depth=5)
    dtc_bin.fit(breast_cancer_data_pandas.x_train, breast_cancer_data_pandas.y_train)

    logger.info(f'Learned tree:\n{dtc_bin.show_tree()}')

    logger.info("Creating general version")
    dtc_gen = decision_tree_classifier(max_depth=5)
    dtc_gen.fit(breast_cancer_data_pandas.x_train, breast_cancer_data_pandas.y_train)

    logger.info(f'Learned tree:\n{dtc_gen.show_tree()}')

    logger.info('Running predictions')
    bin_prob = dtc_bin.predict_proba(breast_cancer_data_pandas.x_test)
    bin_lab = dtc_bin.predict(breast_cancer_data_pandas.x_test)
    gen_prob = dtc_gen.predict_proba(breast_cancer_data_pandas.x_test)
    gen_lab = dtc_gen.predict(breast_cancer_data_pandas.x_test)
    

    logger.info("Comparing")
    assert_allclose(bin_prob, gen_prob)
    assert_allclose(bin_lab, gen_lab)

    logger.info("Done")


def test_dtc_json(breast_cancer_data_pandas, caplog):
    import logging
    import pandas as pd
    from generalizedtrees.recipes import binary_decision_tree_classifier
    from generalizedtrees.vis import explanation_to_JSON
    from generalizedtrees.vis.vis import explanation_to_simplified

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = binary_decision_tree_classifier(max_depth = 5)

    logger.info("Fitting tree")

    dtc.fit(breast_cancer_data_pandas.x_train, breast_cancer_data_pandas.y_train)

    logger.info(f'Learned tree:\n{dtc.show_tree()}')

    annotation = pd.DataFrame({'names': breast_cancer_data_pandas.x_train.columns})

    logger.info(f'Simplified: {explanation_to_simplified(dtc, annotation)}')

    logger.info(f'JSON: {explanation_to_JSON(dtc, annotation)}')
