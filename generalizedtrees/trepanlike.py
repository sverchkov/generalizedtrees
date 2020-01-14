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
from sklearn.base import ClassifierMixin
from generalizedtrees.core import AbstractTreeEstimator
from generalizedtrees.leaves import SimplePredictor

from generalizedtrees.sampling import rejection_sample_generator
from generalizedtrees.scores import gini
from generalizedtrees import splitting
from scipy.stats import mode
from numpy import shape

logger = logging.getLogger()


def make_trepanlike_classifier(
        classifier,
        splitting_strategies=None,
        generator=None,
        constrained_generator=None):
    """
    Make a Trepan-like mimic tree classifier
    :param classifier:
    :param splitting_strategies: Describes how to split each feature (None (the default) is a `<=` vs `>` split)
    :param generator:
    :param constrained_generator:
    :return: Class object whose instances are mimic trees
    """

    if generator is None:
        if constrained_generator is None:
            raise ValueError()
    else:
        constrained_generator = rejection_sample_generator(generator)

    ## Establish splitting strategies:

    if splitting_strategies is None:
        def splitting_strategies(feature, data):
            return splitting.binary_threshold(data, feature)

    if isinstance(splitting_strategies, list):
        splitting_strategies = splitting.compose_splitting_strategies(splitting_strategies)

    class TrepanLikeClassifier(ClassifierMixin, AbstractTreeEstimator):

        def __init__(self, s_min=20, max_depth=5, score=gini):
            self.s_min = s_min
            self.max_depth = max_depth
            self.score = score
            super().__init__()

        def oracle_sample(self, constraints, n=None):
            """
            Generate samples using the oracle
            :param constraints: The cumulative set of constraints at this point in the tree
            :param n: The number of samples to generate
            :return: A tuple of data (n-by-features numpy array) and oracle predictions (length n vector)
            """

            if n is None:
                n = self.s_min

            data = constrained_generator(n, constraints)

            return data, classifier.predict(data)

        def best_split(self, constraints):

            best_split = []

            if len(constraints) < self.max_depth:

                features, targets = self.oracle_sample(constraints)

                best_score = self.score(targets)

                for feature in range(shape(features)[1]):

                    for branches in splitting_strategies(feature, features):

                        scores = [self.score(self.oracle_sample(constraints+(branch,))[1]) for branch in branches]

                        candidate_score = sum(scores)/len(branches)

                        if candidate_score < best_score:
                            best_score = candidate_score
                            best_split = branches

            return best_split

        def leaf_predictor(self, constraints):
            _, targets = self.oracle_sample(constraints)
            return SimplePredictor(mode(targets)[0])

        def check_data_for_predict(self, data):
            # Current implementation is no-op
            return data

    return TrepanLikeClassifier

