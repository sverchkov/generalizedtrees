# Functions implementing classification trees
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

from typing import TypeVar, Protocol, Sized, Iterable
import numpy as np
import pandas as pd
from generalizedtrees.tree import Tree


T = TypeVar('T')


class SizedIterable(Sized, Iterable[T], Protocol[T]):
    pass


def _predict_subtree_proba(node: Tree.Node, data_frame, idx, result_matrix):
    """
    Workhorse for predicting probabilities according to the built tree.

    node: tree node whose item must implement 'node_proba'
    data_frame: a pandas(-like) dataframe
    idx: a numpy array of numeric instance indexes
    result_matrix: a numpy matrix of probabilities
    """

    if node.is_leaf:
        result_matrix[idx,:] = node.item.node_proba(data_frame.iloc[idx])
    
    else:
        branches = node.item.split.pick_branches(data_frame.iloc[idx])
        for b in np.unique(branches):
            _predict_subtree_proba(node[b], data_frame, idx[branches==b], result_matrix)

    return result_matrix


def predict_proba(tree: Tree, data_matrix, target_classes: SizedIterable):

    if not isinstance(data_matrix, pd.DataFrame):
        data_matrix = pd.DataFrame(data_matrix)

    n = data_matrix.shape[0]
    k = len(target_classes)

    result = np.empty((n, k), dtype=np.float)

    return pd.DataFrame(
        _predict_subtree_proba(
            tree.node('root'),
            data_matrix,
            np.arange(n, dtype=np.intp),
            result),
        columns=target_classes,
        copy=False)


def predict(tree: Tree, data_frame, target_classes: SizedIterable):
    return predict_proba(tree, data_frame, target_classes).idxmax(axis=1)


class TreeClassifierMixin:

    def __init__(self):
        self.tree: Tree
        self.target_classes: SizedIterable

    def predict_proba(self, data_frame):
        predict_proba(self.tree, data_frame, self.target_classes)
    
    def predict(self, data_frame):
        predict(self.tree, data_frame, self.target_classes)