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
from generalizedtrees.core import AbstractTreeClassifier
from generalizedtrees.leaves import SimplePredictor
from generalizedtrees.constraints import LEQConstraint, GTConstraint
from generalizedtrees.sampling import gaussian_rejection_sample
from generalizedtrees.scores import gini
from scipy.stats import mode
import numpy as np

logger = logging.getLogger()


class TrepanLike(AbstractTreeClassifier):

    def __init__(self, classifier, s_min=20, score=gini):
        self.feature_means = None
        self.feature_sigmas = None
        self.classifier = classifier
        self.s_min = s_min
        self.score = score
        super().__init__()

    def oracle_sample(self, constraints, n=None):

        if n is None:
            n = self.s_min

        data = gaussian_rejection_sample(self.feature_means, self.feature_sigmas, n, constraints)

        return data, self.classifier.predict(data)

    def best_split(self, constraints):
        features, targets = self.oracle_sample(constraints)

        best_score = self.score(targets)

        best_split = []

        for x_i in features:
            for feature in range(len(x_i)):
                branches = [LEQConstraint(feature, x_i[feature]), GTConstraint(feature, x_i[feature])]

                scores = [self.score(self.oracle_sample(constraints+(branch,))[1]) for branch in branches]

                candidate_score = sum(scores)/len(branches)

                if candidate_score < best_score:
                    best_score = candidate_score
                    best_split = branches

        logger.log(5, best_split)

        return best_split

    def leaf_predictor(self, constraints):
        _, targets = self.oracle_sample(constraints)
        return SimplePredictor(mode(targets)[0])

    def fit(self, data, targets):
        self.feature_means = np.mean(data, axis=0)
        self.feature_sigmas = np.std(data, axis=0)
        self.build()


if __name__ == "__main__":

    from generalizedtrees.evaluations import eval_mimic_on_iris

    logging.basicConfig(level=5)

    eval_mimic_on_iris(TrepanLike)
