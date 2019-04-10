# Discrete Classifier Mimic Tree Model 1
#
# Copyright 2019 Yuriy Sverchkov 2019
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

import generalized_tree_models as gtm
import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import mode
from sklearn.metrics import classification_report
import logging


logger = logging.getLogger(__name__)


def constrained_gaussian_sample(means, cov, constraints, quorum=20):
    logger.debug(f"constraints: {constraints}")
    x = []
    while len(x) < quorum:
        x_new = multivariate_normal(means, cov, quorum-len(x))
        keep = [sample for sample in x_new if all([c.test(sample) for c in constraints])]
        x.extend(keep)
    return x


def gaussian_feature_sampler_leaf_model(means, cov, classifier, constraints, quorum=20):
    x = constrained_gaussian_sample(means, cov, constraints, quorum)
    m, n = mode(classifier.predict(x))
    return gtm.SimplePredictor(m[0])


def gini(y):
    n = len(y)

    if n == 0:
        return 0

    p = np.bincount(y)/n

    return 1 - sum(p*p)


def gaussian_feature_gini_split_function(means, cov, classifier, constraints, quorum=20):
    parent_sample = constrained_gaussian_sample(means, cov, constraints, quorum)
    best_gini = gini(classifier.predict(parent_sample))

    best_split = []

    for x in parent_sample:
        for feature in range(len(x)):
            left_constraint = gtm.LEQConstraint(feature, x[feature])
            right_constraint = ~left_constraint

            lcs = constraints.copy()
            lcs.append(left_constraint)

            left_gini = gini(classifier.predict(constrained_gaussian_sample(means, cov, lcs, quorum)))

            rcs = constraints.copy()
            rcs.append(right_constraint)

            right_gini = gini(classifier.predict(constrained_gaussian_sample(means, cov, rcs, quorum)))

            candidate_gini = left_gini + right_gini

            if candidate_gini > best_gini:
                best_gini = candidate_gini
                best_split = [left_constraint, right_constraint]

    return best_split


if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.svm import SVC

    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    # Load iris
    iris = datasets.load_iris()

    # Learn classifier
    classifier = SVC()
    classifier.fit(iris.data, iris.target)

    # Report SVC classification quality on training set
    print(classification_report(iris.target,
                                classifier.predict(iris.data)))

    iris_mean = np.mean(iris.data, axis=0)
    iris_cov = np.cov(iris.data, rowvar=False)

    # Specify split function
    def split_function(constraints):
        return gaussian_feature_gini_split_function(iris_mean, iris_cov, classifier, constraints)

    # Specify leaf model
    def leaf_model_factory(constraints):
        return gaussian_feature_sampler_leaf_model(iris_mean, iris_cov, classifier, constraints)

    # Define tree
    tree = gtm.GeneralTreeClassifier(split_function, leaf_model_factory)

    # Build tree
    tree.build()

    # Report tree quality
    print(classification_report(iris.target,
                                tree.predict(iris.data)))
