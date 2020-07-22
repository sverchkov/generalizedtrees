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

def test_composed_dtc(breast_cancer_data, caplog):

    import logging
    from generalizedtrees.classifiers import DecisionTreeClassifier
    from generalizedtrees.core import FeatureSpec

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    logger.info("Creating class instance")
    dtc = DecisionTreeClassifier(max_tree_size = 5)

    logger.info("Fitting tree")
    d = breast_cancer_data.x_train.shape[1]

    dtc.fit(breast_cancer_data.x_train, breast_cancer_data.y_train, (FeatureSpec.DISCRETE,)*d)

    logger.info("Running prediction")
    dtc.predict(breast_cancer_data.x_test)

    logger.info("Done")