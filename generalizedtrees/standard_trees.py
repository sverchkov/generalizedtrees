# Decision Tree Classifier Model
#
# This uses the generalized tree model to implement a standard decision tree learner.
# This is mostly an exercise in showing that the framework encompasses standard decision trees.
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
import generalizedtrees.constraints
import generalizedtrees.leaves
from generalizedtrees import core
from scipy.stats import mode
import logging
from generalizedtrees.scores import gini, entropy

logger = logging.getLogger(__name__)


class DecisionTreeClassifier(core.AbstractTreeClassifier):

    def __init__(self, score):
        assert callable(score)
        self.score = score
        self.features = None
        self.targets = None
        super().__init__()

    def best_split(self, constraints):
        index = list(map(core.test_all_x(constraints), self.features))
        features = self.features[index]
        targets = self.targets[index]

        best_score = self.score(targets)

        best_split = []

        for x_i in features:
            for feature in range(len(x_i)):
                left_constraint = generalizedtrees.constraints.LEQConstraint(feature, x_i[feature])
                right_constraint = ~left_constraint

                left = list(map(left_constraint.test, features))
                right = list(map(right_constraint.test, features))

                left_score = self.score(targets[left])

                left_weight = sum(left) / len(targets)

                right_score = self.score(targets[right])

                candidate_score = left_score * left_weight + right_score * (1 - left_weight)

                if candidate_score < best_score:
                    best_score = candidate_score
                    best_split = [left_constraint, right_constraint]

        logger.log(5, best_split)

        return best_split

    def leaf_predictor(self, constraints):
        index = list(map(core.test_all_x(constraints), self.features))
        the_mode, _ = mode(self.targets[index])
        return generalizedtrees.leaves.SimplePredictor(the_mode[0])

    def fit(self, features, targets):
        self.features = features
        self.targets = targets
        self.build()


if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier as SKLDTC
    from sklearn.metrics import classification_report

    logging.basicConfig(level=5)

    # Load iris
    iris = datasets.load_iris()

    # Learn classifiers and report classification quality
    svc = SVC()
    svc.fit(iris.data, iris.target)

    print("Scikit-learn's RBF SVC:")
    print(classification_report(iris.target,
                                svc.predict(iris.data)))

    sk_gini_tree = SKLDTC(criterion='gini')
    sk_gini_tree.fit(iris.data, iris.target)

    print("Scikit-learn's gini decision tree:")
    print(classification_report(iris.target,
                                sk_gini_tree.predict(iris.data)))

    sk_entropy_tree = SKLDTC(criterion='entropy')

    sk_entropy_tree.fit(iris.data, iris.target)

    print("Scikit-learn's entropy decision tree:")
    print(classification_report(iris.target,
                                sk_entropy_tree.predict(iris.data)))

    # Define trees
    tree_g = DecisionTreeClassifier(gini)
    tree_e = DecisionTreeClassifier(entropy)

    # Build trees
    tree_g.fit(iris.data, iris.target)
    logger.debug(tree_g)

    tree_e.fit(iris.data, iris.target)
    logger.debug(tree_e)

    # Report tree quality
    print(classification_report(iris.target,
                                tree_g.predict(iris.data)))
    print(classification_report(iris.target,
                                tree_e.predict(iris.data)))
