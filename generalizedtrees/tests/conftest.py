# Definitions for shared test fixtures
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

from numpy.random import seed
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import namedtuple

@pytest.fixture(scope="module")
def breast_cancer_data():

    # Housekeeping
    seed(20200617)

    # Load data
    bc = load_breast_cancer()

    x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target)

    Dataset = namedtuple('Dataset', ['x_train', 'x_test', 'y_train', 'y_test'])

    return Dataset(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

@pytest.fixture(scope="module")
def breast_cancer_rf_model(breast_cancer_data):

    # Learn 'black-box' model
    rf = RandomForestClassifier()
    rf.fit(breast_cancer_data.x_train, breast_cancer_data.y_train)

    return rf