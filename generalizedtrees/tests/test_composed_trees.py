# Tests for composed tree learners
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

import pytest
from numpy.testing import assert_allclose

def test_composed_dtc(breast_cancer_data, caplog):

    import logging
    from generalizedtrees.recipes import DecisionTreeClassifier
    from generalizedtrees.features import FeatureSpec

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = DecisionTreeClassifier(max_depth = 5)

    logger.info("Fitting tree")
    d = breast_cancer_data.x_train.shape[1]

    dtc.fit(breast_cancer_data.x_train, breast_cancer_data.y_train, feature_spec=(FeatureSpec.CONTINUOUS,)*d)

    logger.info(f'Learned tree:\n{dtc.show_tree()}')

    logger.info("Running prediction")
    dtc.predict(breast_cancer_data.x_test)

    logger.info("Done")


@pytest.mark.skip(reason="Don't know all the details to sklearn's implementation")
def test_composed_dtc_prediction(breast_cancer_data, caplog):
    import logging
    from generalizedtrees.recipes import DecisionTreeClassifier
    from generalizedtrees.features import FeatureSpec
    from sklearn.tree import DecisionTreeClassifier as SKDTC

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = DecisionTreeClassifier(max_depth = 5)

    logger.info("Fitting tree")
    d = breast_cancer_data.x_train.shape[1]

    dtc.fit(breast_cancer_data.x_train, breast_cancer_data.y_train, feature_spec=(FeatureSpec.CONTINUOUS,)*d)

    logger.info(f'Learned tree:\n{dtc.show_tree()}')

    logger.info("Running prediction")
    my_pr = dtc.predict(breast_cancer_data.x_test)

    logger.info("Running Scikit-Learn's DT")
    sk_dtc = SKDTC(criterion='entropy', max_depth=5)
    sk_dtc.fit(breast_cancer_data.x_train, breast_cancer_data.y_train)
    sk_pr = sk_dtc.predict(breast_cancer_data.x_test)

    logger.info("Comparing")
    assert_allclose(my_pr, sk_pr)

    logger.info("Done")


def test_composed_dtc_pandas(breast_cancer_data_pandas, caplog):
    import logging
    from generalizedtrees.recipes import DecisionTreeClassifier

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = DecisionTreeClassifier(max_depth = 5)

    logger.info("Fitting tree")

    dtc.fit(breast_cancer_data_pandas.x_train, breast_cancer_data_pandas.y_train)

    logger.info(f'Learned tree:\n{dtc.show_tree()}')

    logger.info("Running prediction")
    dtc.predict(breast_cancer_data_pandas.x_test)

    logger.info("Done")


def test_composed_trepan_numpy(breast_cancer_data, breast_cancer_rf_model, caplog):

    from generalizedtrees.recipes import Trepan
    from generalizedtrees.features import FeatureSpec
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
    trepan = Trepan()

    logger.info("Fitting tree")
    trepan.fit(x_train, model.predict_proba, feature_spec = (FeatureSpec.CONTINUOUS,)*d)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    logger.info(f'Learned tree:\n{trepan.show_tree()}')

    # Make predictions
    logger.info("Running prediction")
    #trepan_predictions =
    trepan.predict(x_test)

    logger.info("Done")


def test_composed_trepan_pandas(breast_cancer_data_pandas, breast_cancer_rf_model, caplog):

    from generalizedtrees.recipes import Trepan
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
    trepan = Trepan()

    logger.info('Fitting tree')
    oracle = lambda x: pd.DataFrame(model.predict_proba(x), columns=target_names)
    trepan.fit(x_train, oracle)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    logger.info(f'Learned tree:\n{trepan.show_tree()}')

    # Make predictions
    logger.info('Running prediction')
    trepan_predictions = trepan.predict(x_test)

    logger.info(f'Predictions: {list(trepan_predictions)}')

    logger.info("Done")


@pytest.mark.skip(reason="Composed classes are not picklable in current version")
def test_composed_dtc_serialization(breast_cancer_data, caplog):

    import logging
    from generalizedtrees.recipes import DecisionTreeClassifier
    from generalizedtrees.features import FeatureSpec
    from pickle import dumps, loads

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = DecisionTreeClassifier(max_depth = 5)

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


def test_composed_dtc_tree_only_serialization(breast_cancer_data_pandas, caplog):

    import logging
    from generalizedtrees.recipes import DecisionTreeClassifier
    from generalizedtrees.tree import tree_to_str
    from pickle import dumps, loads

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = DecisionTreeClassifier(max_depth = 5)

    logger.info("Fitting tree")

    dtc.fit(breast_cancer_data_pandas.x_train, breast_cancer_data_pandas.y_train)

    tree_str = dtc.show_tree()
    logger.info(f'Learned tree:\n{tree_str}')

    logger.info('Pickling tree')
    bytes_obj = dumps(dtc.tree)

    logger.info('Unpickling tree')
    returned_tree = loads(bytes_obj)

    returned_tree_str = tree_to_str(returned_tree)
    logger.info(f'Unpickled tree:\n{returned_tree_str}')

    assert returned_tree_str == tree_str

    logger.info("Done")


@pytest.mark.skip(reason="Current Trepan uses lambdas which are not picklable")
def test_trepan_tree_only_serialization(breast_cancer_data_pandas, breast_cancer_rf_model, caplog):

    import logging
    from generalizedtrees.recipes import Trepan
    import pandas as pd
    from generalizedtrees.tree import tree_to_str
    from pickle import dumps, loads
    
    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    x_train = breast_cancer_data_pandas.x_train
    x_test = breast_cancer_data_pandas.x_test
    model = breast_cancer_rf_model
    target_names = breast_cancer_data_pandas.target_names

    # Learn explanation
    logger.info('Creating class instance')
    trepan = Trepan()

    logger.info('Fitting tree')
    oracle = lambda x: pd.DataFrame(model.predict_proba(x), columns=target_names)
    trepan.fit(x_train, oracle)

    tree_str = trepan.show_tree()
    logger.info(f'Learned tree:\n{tree_str}')

    logger.info('Pickling tree')
    bytes_obj = dumps(trepan.tree)

    logger.info('Unpickling tree')
    returned_tree = loads(bytes_obj)

    returned_tree_str = tree_to_str(returned_tree)
    logger.info(f'Unpickled tree:\n{returned_tree_str}')

    assert returned_tree_str == tree_str

    logger.info("Done")