# Tests for Trepan
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
from generalizedtrees.explainers import Trepan

@pytest.fixture
def breast_cancer_data_and_model():

    from numpy.random import seed
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from collections import namedtuple

    # Housekeeping
    seed(20200617)

    # Load data
    bc = load_breast_cancer()

    x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target)

    # Learn 'black-box' model
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    Dataset = namedtuple('Dataset', ['x_train', 'x_test', 'y_train', 'y_test'])
    TestBundle = namedtuple('TestBundle', ['data', 'model'])
    return TestBundle(
        data=Dataset(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test),
        model=rf)

def test_trepan(breast_cancer_data_and_model, caplog):

    from generalizedtrees.core import FeatureSpec
    from logging import DEBUG

    caplog.set_level(DEBUG)

    x_train = breast_cancer_data_and_model.data.x_train
    x_test = breast_cancer_data_and_model.data.x_test
    model = breast_cancer_data_and_model.model

    # Learn explanation
    d = x_train.shape[1]

    trepan = Trepan(use_m_of_n = False)
    trepan.fit(x_train, model.predict, (FeatureSpec.CONTINUOUS,)*d)

    # Make predictions
    trepan_predictions = trepan.predict(x_test)

    # TODO: value checking


if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=5)

    test_trepan(breast_cancer_data_and_model())