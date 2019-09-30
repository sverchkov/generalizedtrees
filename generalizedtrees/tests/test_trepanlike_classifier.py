# Tests for standard trees
#
# Copyright 2019 Yuriy Sverchkov
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

from generalizedtrees.trepanlike import make_trepanlike_classifier
from sklearn.utils.estimator_checks import check_estimator
from numpy import random
from sklearn import datasets
from sklearn.svm import SVC
import pytest


@pytest.mark.slow
def test_trepanlike_classifier_with_iris():

    # Load Iris
    iris = datasets.load_iris()

    # Learn classifier
    svc = SVC()
    svc.fit(iris.data, iris.target)

    # Make mimic tree class
    TLC = make_trepanlike_classifier(classifier=svc, generator=lambda n: random.uniform(low=0, high=10, size=(n, 4)))

    # Make mimic tree instance
    tlc = TLC()

    # Build the tree
    tlc.build()

    # Get some predictions
    print(tlc.predict(iris.data))


if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=5)

    test_trepanlike_classifier_with_iris()

