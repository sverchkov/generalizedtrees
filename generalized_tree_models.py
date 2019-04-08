# Generalized Tree Models
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

from __future__ import annotations
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List
from abc import ABC, abstractmethod
from collections import deque
import logging


class Constraint(ABC):

    @abstractmethod
    def test(self, sample):
        pass


class Node:

    def __init__(self, constraint: Constraint = None, parent: Node = None):
        self.parent = parent
        self.constraint = constraint
        self.model = None

    @property
    def all_constraints(self):
        if self.parent:
            return self.parent.all_constraints.append(self.constraint)
        elif self.constraint:
            return [self.constraint]
        else:
            return []


class ChildSelector:

    def __init__(self, children: List[Node]):
        self.children = children

    def predict(self, sample):
        for c in self.children:  # Child constraints should be mutex so we return the first satisfying one
            if c.constraint.test(sample):
                return c.model.predict(sample)
        return None  # Maybe throw exception here?


class NodeQueue:

    def __init__(self):
        self.q = deque()

    def insert(self, x):
        self.q.append(x)

    def insert_all(self, xs):
        self.q.extend(xs)

    def pop(self):
        return self.q.popleft()


class GeneralTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            best_split_function,
            leaf_predictor_factory,
            sequential_access_data_structure_factory=NodeQueue):
        """
        The general tree classifier is defined by
        :param best_split_function: Function that determines the best next split given a list of constraints. This
        function also implicitly defines the type of constraint used throughout the tree via its return method.
        :param leaf_predictor_factory: A factory for the leaf predictor (e.g. linear model for a linear model tree or
        a class prediction for a standard decision tree) Leaf predictors must implement predict(sample)
        :param sequential_access_data_structure_factory: A  factory that produces a data structure such as a stack,
        queue, or priority queue that determines the order in which the tree is built. For a priority queue the rule for
        comparing nodes is also communicated through this function. The data structure must implement insert (for one
        element), insert_all (for a list of elements) and pop.
        """
        assert callable(best_split_function), "Best split function must be callable"
        assert callable(leaf_predictor_factory), "Leaf predictor factory must be callable"
        assert callable(sequential_access_data_structure_factory),\
            "Sequential access data structure factory must be callable"
        self.root = None
        self.best_split_function = best_split_function
        self.leaf_predictor_factory = leaf_predictor_factory
        self.sequential_access_data_structure_factory = sequential_access_data_structure_factory

    def best_split(self, node: Node):
        return self.best_split_function(node.all_constraints)

    def leaf_predictor(self, node: Node):
        return self.leaf_predictor_factory(node)

    def build(self):
        self.root = Node()
        nodes = self.sequential_access_data_structure_factory()
        nodes.insert(self.root)

        while nodes:
            parent = nodes.pop()
            split = self.best_split(parent)

            if len(split) > 1:  # Splitting
                children = [Node(s, parent) for s in split]
                parent.model = ChildSelector(children)
                nodes.insert_all(children)

            else:  # It's a leaf
                parent.model = self.leaf_predictor(parent)

    def predict(self, data):
        [self.predict_instance(x) for x in data]

    def predict_instance(self, sample):
        self.root.model.predict(sample)

