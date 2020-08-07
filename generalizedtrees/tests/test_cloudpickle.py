# Serialization tests with cloudpickle
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
try:
    import cloudpickle as cp
    no_cloudpickle = False
except:
    no_cloudpickle = True


@pytest.mark.skipif(no_cloudpickle, reason='Skipping cloudpickle tests (could not load cloudpickle)')
def test_trepan_cloudpickle_serialization(breast_cancer_data_pandas, breast_cancer_rf_model, caplog):

    import logging
    from generalizedtrees.recipes import Trepan
    import pandas as pd
    from generalizedtrees.tree import tree_to_str
    
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

    logger.info('Pickling internal tree')
    bytes_obj = cp.dumps(trepan.tree)

    logger.info('Unpickling internal tree')
    returned_internal_tree = cp.loads(bytes_obj)

    returned_internal_tree_str = tree_to_str(returned_internal_tree)
    logger.info(f'Unpickled internal tree:\n{returned_internal_tree_str}')

    assert returned_internal_tree_str == tree_str

    logger.info('Pickling Trepan instance')
    bytes_obj = cp.dumps(trepan)

    logger.info('Unpickling Trepan instance')
    returned_tree = cp.loads(bytes_obj)

    returned_tree_str = returned_tree.show_tree()
    logger.info(f'Unpickled tree:\n{returned_tree_str}')

    assert returned_tree_str == tree_str

    logger.info("Done")


@pytest.mark.skipif(no_cloudpickle, reason='Skipping cloudpickle tests (could not load cloudpickle)')
def test_composed_dtc_cloudpickle_serialization(breast_cancer_data, caplog):

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

    tree_str = dtc.show_tree()
    logger.info(f'Learned tree:\n{tree_str}')

    logger.info('Pickling tree')
    bytes_obj = cp.dumps(dtc)

    logger.info('Unpickling tree')
    returned_dtc = cp.loads(bytes_obj)

    returned_tree_str = returned_dtc.show_tree()
    logger.info(f'Unpickled tree:\n{returned_tree_str}')

    assert returned_tree_str == tree_str

    logger.info("Running unpicled tree prediction")
    returned_dtc.predict(breast_cancer_data.x_test)

    logger.info("Done")