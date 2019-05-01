# Evaluation suites
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


logger = logging.getLogger()


# Version that fits the mimic on the same training data as the classifier
def eval_mimic_on_iris(mimic_tree_class):

    # Data part
    iris = datasets.load_iris()
    data = iris.data
    y = iris.target

    for train_index, test_index in StratifiedShuffleSplit(n_splits=1, test_size=0.4).split(data, y):

        # Learn a model with train
        classifier = SVC()
        classifier.fit(data[train_index], y[train_index])

        # Learn our tree using a model and test set
        tree = mimic_tree_class(classifier)
        tree.fit(data[train_index], y[train_index])

        # Learn a standard tree from the train+test sets
        skl_tree = DecisionTreeClassifier()
        skl_tree.fit(data[train_index], y[train_index])

        # Evaluate on validation set
        print("RBF SVC Classifier:")
        print(classification_report(y[test_index],
                                    classifier.predict(data[test_index])))

        print("Scikit-learn Decision Tree:")
        print(classification_report(y[test_index],
                                    skl_tree.predict(data[test_index])))

        print("Mimic tree:")
        print(tree)
        print(classification_report(y[test_index],
                                    tree.predict(data[test_index])))
