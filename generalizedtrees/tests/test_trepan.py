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

def test_trepan(breast_cancer_data, breast_cancer_rf_model, caplog):

    from generalizedtrees.core import FeatureSpec
    from logging import DEBUG

    caplog.set_level(DEBUG)

    x_train = breast_cancer_data.x_train
    x_test = breast_cancer_data.x_test
    model = breast_cancer_rf_model

    # Learn explanation
    d = x_train.shape[1]

    trepan = Trepan(use_m_of_n = False)
    trepan.fit(x_train, model.predict, (FeatureSpec.CONTINUOUS,)*d)

    # Make predictions
    trepan_predictions = trepan.predict(x_test)

    # TODO: value checking
