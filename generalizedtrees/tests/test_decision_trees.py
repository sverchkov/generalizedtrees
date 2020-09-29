# Tests for regular classification trees
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

def test_dtc(breast_cancer_data, caplog):

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
def test_dtc_prediction(breast_cancer_data, caplog):
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


def test_dtc_pandas(breast_cancer_data_pandas, caplog):
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

def test_dtc_json(breast_cancer_data_pandas, caplog):
    import logging
    import pandas as pd
    from generalizedtrees.recipes import DecisionTreeClassifier
    from generalizedtrees.vis import explanation_to_JSON

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = DecisionTreeClassifier(max_depth = 5)

    logger.info("Fitting tree")

    dtc.fit(breast_cancer_data_pandas.x_train, breast_cancer_data_pandas.y_train)

    logger.info(f'Learned tree:\n{dtc.show_tree()}')

    annotation = pd.DataFrame({'names': breast_cancer_data_pandas.x_train.columns})

    logger.info(f'JSON: {explanation_to_JSON(dtc, annotation)}')
