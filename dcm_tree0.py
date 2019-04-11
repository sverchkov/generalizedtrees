# Classifier Tree Model 0
#
# This uses the generalized tree model to implement a standard decision tree learner using the Gini criterion
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
from typing import Tuple
import logging


logger = logging.getLogger(__name__)


def select(x, y, constraints: Tuple):
    tuples = [(x_i, y_i) for (x_i, y_i) in zip(x, y) if all([c.test(x_i) for c in constraints])]
    if tuples:
        return zip(*tuples)
    else:
        return [], []


def training_set_leaf_model(x, y, constraints: Tuple):
    _, y_subset = select(x, y, constraints)
    the_mode, _ = mode(y_subset)
    return gtm.SimplePredictor(the_mode[0])


def gini(y):
    n = len(y)

    if n == 0:
        return 0

    p = np.bincount(y)/n

    return 1 - sum(p*p)


def entropy(y):
    n = len(y)

    if n == 0:
        return 0

    p = np.bincount(y)/n
    pl2p = np.where(p > 0, -p*np.log2(p), 0)

    return sum(pl2p)


def training_set_best_split_function(x, y, constraints: Tuple, score):
    parent_x, parent_y = select(x, y, constraints)
    best_score = score(parent_y)

    best_split = []

    for x_i in parent_x:
        for feature in range(len(x_i)):
            left_constraint = gtm.LEQConstraint(feature, x_i[feature])
            right_constraint = ~left_constraint

            _, left_y = select(parent_x, parent_y, (left_constraint,))

            left_score = score(left_y)

            left_weight = len(left_y)/len(parent_y)

            _, right_y = select(parent_x, parent_y, (right_constraint,))

            right_score = score(right_y)

            candidate_score = left_score * left_weight + right_score * (1-left_weight)

            if candidate_score < best_score:
                best_score = candidate_score
                best_split = [left_constraint, right_constraint]

    logger.log(5, best_split)

    return best_split


if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    logging.basicConfig(level=5)

    # Load iris
    iris = datasets.load_iris()

    # Learn classifiers and report classification quality
    svc = SVC()
    svc.fit(iris.data, iris.target)

    print("Scikit-learn's RBF SVC:")
    print(classification_report(iris.target,
                                svc.predict(iris.data)))

    sk_gini_tree = DecisionTreeClassifier(criterion='gini')
    sk_gini_tree.fit(iris.data, iris.target)

    print("Scikit-learn's gini decision tree:")
    print(classification_report(iris.target,
                                sk_gini_tree.predict(iris.data)))

    sk_entropy_tree = DecisionTreeClassifier(criterion='entropy')

    sk_entropy_tree.fit(iris.data, iris.target)

    print("Scikit-learn's entropy decision tree:")
    print(classification_report(iris.target,
                                sk_entropy_tree.predict(iris.data)))

    # Specify split functions
    def split_function(constraints):
        return training_set_best_split_function(iris.data, iris.target, constraints, gini)

    def split_function_e(constraints):
        return training_set_best_split_function(iris.data, iris.target, constraints, entropy)

    # Specify leaf model
    def leaf_model_factory(constraints):
        return training_set_leaf_model(iris.data, iris.target, constraints)

    # Define trees
    tree_g = gtm.GeneralTreeClassifier(split_function, leaf_model_factory)
    tree_e = gtm.GeneralTreeClassifier(split_function_e, leaf_model_factory)

    # Build trees
    tree_g.build()
    logger.debug(tree_g)

    tree_e.build()
    logger.debug(tree_e)

    # Report tree quality
    print(classification_report(iris.target,
                                tree_g.predict(iris.data)))
    print(classification_report(iris.target,
                                tree_e.predict(iris.data)))
