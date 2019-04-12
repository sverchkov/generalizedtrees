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

import logging
from generalized_tree_models import GeneralTreeClassifier, SimplePredictor, test_all_x
from scipy.stats import mode


class MimicTreeClassifier(GeneralTreeClassifier):

    def __init__(self, classifier):
        self.test_set = None
        self.y = None
        self.predictions = None
        self.classifier = classifier
        super().__init__(self.loss_split_function, self.leaf_factory)

    def leaf_factory(self, constraints):
        if not self.predictions:
            raise UserWarning("Looks like you tried to build the mimic tree without fitting a test set first!")

        y_hats = [y_i for (X_i, y_i) in self.predictions if test_all_x(constraints)(X_i)]
        if not y_hats:  # No samples falling in constraints, predict most common class
            return SimplePredictor(mode(self.y)[0])
        else:
            return SimplePredictor(mode(y_hats)[0])

    def loss_split_function(self, constraints):
        best_split = []
        return best_split

    def fit(self, features, target):
        self.y = target
        self.test_set = zip(features, target)
        self.predictions = zip(features, self.classifier.predict(features))
        self.build()


if __name__ == "__main__":

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn import datasets
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report

    logging.basicConfig(level=5)

    # General plan for split computation:
    # Given a test set and a learned model
    # At each split, given parent constraint c and l/r split constraints:
    # Compute loss for test set that falls in c with features* [not fixed in c](?) permuted
    # Compute loss for test that talls in c+l with features* permuted
    # Compute loss for test set that falls in c+r with features* permuted
    # Best (or no) split is at best loss(c+l) + loss(c+r) (or loss(c) )

    # Data part
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    n = len(X)

    for train_index, rest in StratifiedShuffleSplit(n_splits=1, test_size=0.4).split(X, y):
        for test_index, validate_index in StratifiedShuffleSplit(n_splits=1, test_size=0.5).split(X[rest], y[rest]):

            # Learn a model with train
            classifier = SVC()
            classifier.fit(X[train_index], y[train_index])

            # Learn our tree using a model and test set
            tree = MimicTreeClassifier(classifier)
            tree.fit(X[test_index], y[test_index])

            # Learn a standard tree from the train+test sets
            skl_tree = DecisionTreeClassifier()
            skl_tree.fit(X[[*train_index, *test_index]], y[[*train_index, *test_index]])

            # Evaluate on validation set
            print("RBF SVC Classifier:")
            print(classification_report(y[validate_index],
                                        classifier.predict(X[validate_index])))

            print("Scikit-learn Decision Tree:")
            print(classification_report(y[validate_index],
                                        skl_tree.predict(X[validate_index])))

            print("Mimic tree:")
            print(tree)
            print(classification_report(y[validate_index],
                                        tree.predict(X[validate_index])))
