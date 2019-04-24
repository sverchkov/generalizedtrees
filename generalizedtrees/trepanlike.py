# Trepan-like mimic tree learner
#
# Based on Trepan (Craven and Shavlik 1996)
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

import logging
from generalizedtrees.core import AbstractTreeClassifier, SimplePredictor
from scipy.stats import mode

logger = logging.getLogger()


class TrepanLike(AbstractTreeClassifier):

    def __init__(self, classifier, s_min=10):
        self.data = None
        self.targets = None
        self.classifier = classifier
        self.s_min = s_min
        super().__init__()

    def oracle_sample(self, constraints, n=None):

        if n is None:
            n = self.s_min

        data = None

        constraints_dict = {}  # constraints_by_feature(constraints)

        for feature, cs in constraints_dict.items():
            pass  # Sample feature subject to constraints

        return data, self.classifier.predict(data)

    def leaf_predictor(self, constraints):
        _, targets = self.oracle_sample(constraints)
        return SimplePredictor(mode(targets)[0])

    def fit(self, data, targets):
        self.data = data
        self.targets = targets
        self.build()


if __name__ == "__main__":

    from generalizedtrees.evaluations import eval_mimic_on_iris

    logging.basicConfig(level=5)

    eval_mimic_on_iris(TrepanLike)
