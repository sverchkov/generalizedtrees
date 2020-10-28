# Functions and mixins implementing decision tree prediction
#
# Copyright 2020 Yuriy Sverchkov
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

import numpy as np
from pandas import DataFrame

from generalizedtrees.tree import Tree


def _estimate_subtree(node: Tree.Node, data_matrix, idx, result_matrix):
    """
    Workhorse for estimation using a built tree.

    node: tree node,
        if it is a leaf, its item must contain a 'model' (Estimator) member;
        if it is internal, its item must contain a 'split' (SplitTest) member
    data_matrix: a numpy(-like) matrix of n instances by m features
    idx: a numpy array of numeric instance indexes
    result_matrix: a numpy matrix of outputs
    """

    if node.is_leaf:
        result_matrix[idx,:] = node.item.model.estimate(data_matrix[idx,:])
    
    else:
        branches = node.item.split.pick_branches(data_matrix[idx,:])
        for b in np.unique(branches):
            _estimate_subtree(node[b], data_matrix, idx[branches==b], result_matrix)

    return result_matrix


def estimate(tree: Tree, data_matrix, target_dimension):

    n = data_matrix.shape[0]

    return _estimate_subtree(
        tree.node('root'),
        data_matrix,
        np.arange(n, dtype=np.intp),
        np.empty(
            (n, target_dimension),
            dtype=np.float))


class BaseTreeClassifierMixin:

    # Base class for tree classifiers, defining predict as a function of predict_proba

    # Required members:
    # self.tree (Tree)
    # self.classes_ (A numpy array of class labels)

    def predict(self, data_matrix, as_labels=True):

        proba = self.predict_proba(data_matrix)
        max_idx = proba.argmax(axis=1)

        if as_labels:
            return self.classes_[max_idx]
        else:
            return max_idx


class TreeClassifierMixin(BaseTreeClassifierMixin):

    # Required members:
    # self.tree (Tree)
    # self.classes_ (A numpy array of class labels)

    def predict_proba(self, data_matrix, as_dataframe=False):

        if isinstance(data_matrix, DataFrame):
            data_matrix = data_matrix.to_numpy()

        proba = estimate(self.tree, data_matrix, len(self.classes_))

        if as_dataframe:
            return DataFrame(proba, columns=self.classes_)
        else:
            return proba


class TreeBinaryClassifierMixin(BaseTreeClassifierMixin):

    # The binary classifier is different from the classifier in that the
    # underlying estimator is 1-dimensional, with the estimate reflecting
    # the probability of the +ve (second) class

    # Required members:
    # self.tree (Tree)
    # self.classes_ (A numpy array of class labels)

    def predict_proba(self, data_matrix, as_dataframe=False):

        if isinstance(data_matrix, DataFrame):
            data_matrix = data_matrix.to_numpy()
        
        p_1 = estimate(self.tree, data_matrix, 1)
        proba = np.concatenate((1 - p_1, p_1), axis=1)

        if as_dataframe:
            return DataFrame(proba, columns=self.classes_)
        else:
            return proba


class TreeRegressorMixin:

    # Required members:
    # self.tree (Tree)
    # self.target_dim (dimension of target vector)
    # TODO: flattening of 1-d target?

    def predict(self, data_matrix):
        return estimate(self.tree, data_matrix, self.target_dim)
