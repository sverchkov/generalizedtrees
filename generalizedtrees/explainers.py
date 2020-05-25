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

from collections import namedtuple
from generalizedtrees.tree import TreeNode, Tree

from sklearn.base import ClassifierMixin
from generalizedtrees.core import AbstractTreeEstimator, Node, ChildSelector
from generalizedtrees.leaves import SimplePredictor

from generalizedtrees.sampling import rejection_sample_generator
from generalizedtrees.scores import gini
from generalizedtrees import splitting
from scipy.stats import mode

from heapq import heappush, heappop
from statistics import mode
import numpy as np

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

                for feature in range(features.shape[2]):

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


class Trepan(): # TODO: class hierarchy?
    """
    Implementation of Trepan (Craven and Shavlik 1996)

    An implementation of Trepan, particularly:
    * Uses m-of-n splits
    * Uses marginal kernel density estimation (with bandwidth sigma=1/sqrt(m) where m is
    the number of samples) for continuous variables
    * Uses the empirical distribution for discrete variables
    * Statistical test used to determine what distribution of X to use at any given branch
    """

    class Node(TreeNode):

        def __init__(self):
            # TODO: Set types
            self.score = None
            self.constraints = None
            self.training_idx = None
            self.generator = None
            self.gen_data = None
            self.gen_targets = None
            self.prediction = None
            self.fidelity = None
            self.coverage = None
            super().__init__()

        # TODO: Add comparator


    def __init__(self, max_tree_size = 10):
        # Note: parameters passed to self sould define *how* the explanation tree is built
        # Note: init should declare all members

        self.tree: Tree
        self.oracle = None # f(x) -> y
        self.train_features = None # training set x (no y's)
        self.feature_spec = None # Array of feature specs
        self.min_sample = None # A parameter

        # Stopping criteria:
        self.max_tree_size = max_tree_size

    def new_generator(self, data):
        """
        Returns a new data generator fit to data.
        """
        raise NotImplementedError # TODO

    def construct_split(self, data, targets):
        raise NotImplementedError # TODO

    def fit(self, data, oracle, max_tree_size = None):
        # Note: parameters passed to fit should represent problem-specific details
        # Max tree size can be seen as problem-specific, so we include it here too.
        if max_tree_size is not None:
            self.max_tree_size = max_tree_size

        self.oracle = oracle

        # Targets of training data
        targets = self.oracle(data)

        # n is the number of samples
        n: int = data.shape[0]

        # Init root node and tree
        root = Trepan.Node()
        self.tree = root.plant_tree()

        # Root node extra data generator
        root.generator = self.new_generator(data) 

        # Root node extra data
        root.gen_data = self.draw_sample((), self.min_sample-n, root.generator)

        # Classify extra data
        root.gen_targets = self.oracle(root.gen_data)

        # Estimate root's prediction
        root.prediction = mode(np.append(targets, root.gen_targets))

        # Estimate root's fidelity
        root.fidelity = np.mean(np.append(targets, root.gen_targets) == root.prediction)

        # Other housekeeping
        root.training_idx = range(n)
        root.coverage = 1
        root.score = 0
        root.constraints = ()

        # Insert into heap
        heap = [root]

        while heap and self.tree.size < self.max_tree_size:

            node = heappop(heap)
            
            split = self.construct_split(
                np.append(data[node.training_idxs], node.gen_data),
                np.append(targets[node.training_idxs], node.gen_targets))
            
            for constraint in split: # TODO: Continue from here

                # Initialize a child node
                child = Trepan.Node()
                node.add_child(child)
                child.constraints = node.constraints + (constraint,)

                # Filter training data that gets to the child node
                child.training_idx = [i for i in node.training_idx if constraint.test(data[i])]

                # Re-estimate generator at child node
                child.generator = self.new_generator(data[child.training_idx])

                # Generate data for child node
                child.gen_data = self.draw_sample(
                    child.constraints,
                    self.min_sample-len(child.training_idx),
                    child.generator)

                # Classify generated data
                child.gen_targets = self.oracle(child.gen_data)

                # Compute child's prediction
                child.prediction = mode(np.append(targets[child.training_idx], child.gen_targets))

                # Estimate child's fidelity
                child.fidelity = np.mean(np.append(targets, child.gen_targets) == child.prediction)

                child.coverage = \
                    (len(child.training_idx) + sum(constraint.test(node.gen_data))) / \
                    (len(node.training_idx) + len(node.gen_data)) * \
                    node.coverage
                
                # Node score is the negative of:
                #   f(N) = reach(N) (1-fidelity(N))
                child.score = -(child.coverage * (1 - child.fidelity))
                
                # TODO: local stopping criteria
                if not all( targets[child.training_idx] == child.prediction):
                    heappush(heap, child)

        return self

    def draw_sample(self, constraints, n, generator):
        # Original implementation draws samples one at a time.
        # May be worth optimizing.
        return [self.draw_instance(constraints, generator) for i in range(max(0,n))]

    def draw_instance(self, constraints, generator):
        pass # TODO. See page 50 of thesis
