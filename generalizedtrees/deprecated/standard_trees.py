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
from sklearn.base import BaseEstimator, ClassifierMixin
from generalizedtrees.core import AbstractTreeEstimator
from generalizedtrees.leaves import SimplePredictor
from generalizedtrees.constraints import GTConstraint, LEQConstraint
from generalizedtrees import core
from scipy.stats import mode
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from numpy import unique, inf, ndarray
import logging
from generalizedtrees.scores import gini, entropy

logger = logging.getLogger(__name__)


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin, AbstractTreeEstimator):

    def __init__(self, split_score=gini, enforce_finite=True):
        assert callable(split_score)
        self.split_score = split_score
        self.enforce_finite = enforce_finite
        self.data_: ndarray
        self.targets_: ndarray
        super().__init__()

    def best_split(self, constraints):
        index = list(map(core.test_all_x(constraints), self.data_))
        features = self.data_[index]
        targets = self.targets_[index]

        best_score = self.split_score(targets)

        best_split = []

        for x_i in features:
            for feature in range(len(x_i)):
                left_constraint = LEQConstraint(feature, x_i[feature])
                right_constraint = ~left_constraint

                left = list(map(left_constraint.test, features))
                right = list(map(right_constraint.test, features))

                left_score = self.split_score(targets[left])

                left_weight = sum(left) / len(targets)

                right_score = self.split_score(targets[right])

                candidate_score = left_score * left_weight + right_score * (1 - left_weight)

                if candidate_score < best_score:
                    best_score = candidate_score
                    best_split = [left_constraint, right_constraint]

        logger.log(5, best_split)

        return best_split

    def leaf_predictor(self, constraints):
        index = list(map(core.test_all_x(constraints), self.data_))
        the_mode, _ = mode(self.targets_[index])
        return SimplePredictor(the_mode[0])

    def fit(self, data: ndarray, y: ndarray):

        self.data_, self.targets_ = check_X_y(data, y)
        check_classification_targets(self.targets_)

        self.classes_ = unique(self.targets_)
        self.m_ = self.data_.shape[1]

        self.build()

        return self

    def check_data_for_predict(self, data):
        checked_data = check_array(data, force_all_finite=self.enforce_finite, estimator=self)
        if checked_data.shape[1] != self.m_:
            raise ValueError((f"The number of features in predict ({checked_data.shape[1]})"
                              f" is different from the number of features in fit ({self.m_})."))
        return checked_data


class ModelTree(BaseEstimator, ClassifierMixin, AbstractTreeEstimator):

    def __init__(self, weak_model, max_depth=5):
        self.weak_model = weak_model
        self.features = None
        self.targets = None
        self.max_depth = max_depth
        super().__init__()

    def _training_loss(self, data, targets):

        y_hat = self._train_leaf_predictor(data, targets).predict(data)
        return sum((targets - y_hat)**2)

    def best_split(self, constraints):

        best_split = []

        if len(constraints) <= self.max_depth:  # Using the fact that the constraint list length is the depth

            index = list(map(core.test_all_x(constraints), self.features))
            features = self.features[index]
            targets = self.targets[index]

            best_score = self._training_loss(features, targets)

            for x_i in features:
                for feature in range(len(x_i)):

                    candidate_split = [LEQConstraint(feature, x_i[feature]), GTConstraint(feature, x_i[feature])]

                    candidate_score = self._score_split(candidate_split, features, targets)

                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_split = candidate_split

        return best_split

    def _score_split(self, candidate_split, features, targets):
        score = 0
        for constraint in candidate_split:
            index = list(map(constraint.test, features))
            if sum(index) < 1:
                return inf
            score += self._training_loss(features[index], targets[index])  # Overfitting?
        return score

    def _train_leaf_predictor(self, features, targets):
        unique_targets = unique(targets)

        if len(unique_targets) == 1:
            return SimplePredictor(unique_targets[0])

        model = clone(self.weak_model)
        model.fit(features, targets)
        return model

    def leaf_predictor(self, constraints):
        index = list(map(core.test_all_x(constraints), self.features))

        return self._train_leaf_predictor(self.features[index], self.targets[index])

    def fit(self, features, targets):
        self.features = features
        self.targets = targets
        self.build()
        return self


if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier as SKLDTC
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression

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
    print("Our gini decision tree:")
    print(classification_report(iris.target,
                                tree_g.predict(iris.data)))

    print("Our entropy decision tree:")
    print(classification_report(iris.target,
                                tree_e.predict(iris.data)))

    logger.info("Learning logistic model tree:")
    model_tree = ModelTree(weak_model=LogisticRegression(), max_depth=1)
    model_tree.fit(iris.data, iris.target)

    print("Our model tree:")
    predictions = model_tree.predict(iris.data)
    logger.debug(predictions)
    print(classification_report(iris.target,
                                predictions))
