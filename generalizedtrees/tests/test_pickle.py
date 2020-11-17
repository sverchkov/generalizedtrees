# Test pickling of learned explanations
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

import pytest

def test_dtc_serialization(breast_cancer_data, caplog):

    import logging
    from generalizedtrees.recipes import binary_decision_tree_classifier
    from generalizedtrees.features import FeatureSpec
    from pickle import dumps, loads

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = binary_decision_tree_classifier(max_depth = 5)

    logger.info("Fitting tree")
    d = breast_cancer_data.x_train.shape[1]

    dtc.fit(breast_cancer_data.x_train, breast_cancer_data.y_train, feature_spec=(FeatureSpec.CONTINUOUS,)*d)

    tree_str = dtc.show_tree()
    logger.info(f'Learned tree:\n{tree_str}')

    logger.info('Pickling tree')
    bytes_obj = dumps(dtc)

    logger.info('Unpickling tree')
    returned_dtc = loads(bytes_obj)

    returned_tree_str = returned_dtc.show_tree()
    logger.info(f'Unpickled tree:\n{returned_tree_str}')

    assert returned_tree_str == tree_str

    logger.info("Running unpicled tree prediction")
    returned_dtc.predict(breast_cancer_data.x_test)

    logger.info("Done")


def test_trepan_serialization(breast_cancer_data_pandas, breast_cancer_rf_model, caplog):

    import logging
    from generalizedtrees.recipes import trepan
    from pickle import dumps, loads
    
    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    x_train = breast_cancer_data_pandas.x_train
    x_test = breast_cancer_data_pandas.x_test
    model = breast_cancer_rf_model

    # Learn explanation
    logger.info('Creating class instance')
    trepan = trepan(max_tree_size = 5)

    logger.info('Fitting tree')
    oracle = model.predict_proba
    trepan.fit(x_train, oracle)

    tree_str = trepan.show_tree()
    logger.info(f'Learned tree:\n{tree_str}')

    logger.info('Pickling tree')
    bytes_obj = dumps(trepan)

    logger.info('Unpickling tree')
    returned_tree = loads(bytes_obj)

    returned_tree_str = returned_tree.show_tree()

    logger.info(f'Unpickled tree:\n{returned_tree_str}')

    assert returned_tree_str == tree_str

    logger.info("Done")