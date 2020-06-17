# Explanation tree learners
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
import numpy as np
from collections import namedtuple
from generalizedtrees import splitting
from generalizedtrees.core import FeatureSpec, AbstractTreeEstimator, Node, ChildSelector
from generalizedtrees.constraints import MofN
from generalizedtrees.leaves import SimplePredictor
from generalizedtrees.sampling import rejection_sample_generator
from generalizedtrees.scores import gini
from generalizedtrees.splitting import fayyad_thresholds, one_vs_all
from generalizedtrees.tree import TreeNode, Tree, tree_to_str
from heapq import heappush, heappop
from scipy.stats import mode, ks_2samp, chisquare
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KernelDensity
from statistics import mode
from typing import Tuple


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

    def __init__(
        self,
        max_tree_size = 10,
        rng: np.random.Generator = np.random.default_rng(),
        dist_test_alpha = 0.05,
        use_m_of_n = True,
        beam_width = 2):
        # Note: parameters passed to init sould define *how* the explanation tree is built
        # Note: init should declare all members

        self.tree: Tree
        self.data: np.ndarray
        self.oracle = None # f(x) -> y
        self.train_features = None # training set x (no y's)
        self.feature_spec: Tuple[FeatureSpec, ...] # Tuple of feature specs
        self.min_sample = None # A parameter
        self.rng = rng
        self.dist_test_alpha = dist_test_alpha
        self.use_m_of_n = use_m_of_n
        self.beam_width = beam_width

        # Inferred values
        self._d: int

        # Stopping criteria:
        self.max_tree_size: int = max_tree_size

    Generator = namedtuple("Generator", ["generate", "training_idx"])

    class Node(TreeNode):

        def __init__(self):
            # TODO: Set types
            self.score = None
            self.local_constraint = None
            self.constraints = None
            self.training_idx = None
            self.generator = None
            self.generator_src_node = None
            self.gen_data = None
            self.gen_targets = None
            self.prediction = None
            self.fidelity = None
            self.coverage = None
            super().__init__()

        # TODO: Add comparator

        def __str__(self):

            if self.depth == 0:
                return 'Root'

            elif self.is_leaf:
                return f'If {self.local_constraint} predict {self.prediction}'
            
            else:
                return f'If {self.local_constraint}'

    def _feature_generator(self, data_vector, feature: FeatureSpec):

        n = len(data_vector)

        if feature is FeatureSpec.CONTINUOUS:
            # Sample from a KDE.
            # We use Generator and not RandomState but KDE implementations use RandomState
            # so it's more reliable to just implement the sampling here. 
            return lambda: self.rng.normal(
                loc = self.rng.choice(data_vector, size=1),
                scale = 1/np.sqrt(n),
                size = 1)

        elif feature & FeatureSpec.DISCRETE:
            # Sample from the empirical distribution
            values, counts = np.unique(data_vector, return_counts=True)
            return lambda: self.rng.choice(values, p=counts/n)
        
        else:
            raise ValueError(f"I don't know how to handle feature spec {feature}")

    def new_generator(self, training_idx):
        """
        Returns a new data generator fit to data.
        """

        # The Trepan generator independently generates the individual feature values.
        feature_generators = [
            self._feature_generator(self.data[training_idx,i], self.feature_spec[i])
            for i in range(self._d)]

        return Trepan.Generator(
            generate = lambda: [f() for f in feature_generators],
            training_idx = training_idx)

    def same_distribution(self, idx_1, idx_2):
        """
        Performs statistical test to determine if data subsets defined by indexes are from the
        same distribution.
        """
        n_tests = 0
        min_p = 1.0
        for i in range(self._d):

            if self.feature_spec[i] & FeatureSpec.DISCRETE:
                # Get frequencies.
                # Note: we're assuming that the union of values present in the samples is
                # the set of possible values. This is not all possible values that the
                # variable could originally take.
                v1, c1 = np.unique(self.data[idx_1, i], return_counts=True)
                map1 = {v1[j]: c1[j] for j in range(len(v1))}

                v2, c2 = np.unique(self.data[idx_2, i], return_counts=True)
                map2 = {v2[j]: c2[j] for j in range(len(v2))}

                values = np.union1d(v1, v2)
                k = len(values)

                # If only one value is present skip this test
                if k > 1:

                    freq1 = [map1[v] if v in map1 else 0 for v in values]
                    freq2 = [map2[v] if v in map2 else 0 for v in values]

                    _, p = chisquare(f_obs=freq1, f_exp=freq2, ddof=k-1)
                    if p < min_p:
                        min_p = p
                    n_tests += 1
            
            else:
                # KS-test
                _, p = ks_2samp(self.data[idx_1, i], self.data[idx_2, i])
                if p < min_p:
                    min_p = p
                n_tests += 1

        return min_p < self.dist_test_alpha/n_tests

    def construct_split(self, data, targets):
        split_candidates = self.make_split_candidates(data, targets)
        best_split = ()
        best_split_score = 0
        for split in split_candidates:
            new_score = self.split_score(split, data, targets)
            if new_score > best_split_score:
                best_split_score = new_score
                best_split = split
        
        # For testing if m-of-n splits are really useful
        if self.use_m_of_n:
            return self.construct_m_of_n_split(
                best_split,
                best_split_score,
                split_candidates,
                data,
                targets)
        else:
            return best_split

    def make_split_candidates(self, data, targets):
        # Note: we could make splits based on original training examples only, or on
        # training examples and generated examples.
        # In the current form, this could be controlled by the calling function.

        result = []

        for j in range(self._d):
            if self.feature_spec[j] is FeatureSpec.CONTINUOUS:
                result.extend(fayyad_thresholds(data, targets, j))
            elif self.feature_spec[j] & FeatureSpec.DISCRETE:
                result.extend(one_vs_all(data, j))
            else:
                raise ValueError(f"I don't know how to handle feature spec {self.feature_spec[j]}")
        
        return result

    def split_score(self, split, data, targets):
        raise NotImplementedError

    def construct_m_of_n_split(
        self,
        best_split,
        best_split_score,
        split_candidates,
        data,
        targets):

        beam = [(best_split_score, best_split)]
        beam_changed = True
        while beam_changed:
            beam_changed = False

            for _, split in beam:
                for new_split in MofN.neighboring_tests(split, split_candidates):
                    if self.tests_sig_diff(split, new_split): #+data?
                        new_score = self.split_score(new_split, data, targets)
                        # Pseudocode in paper didn't have this but it needs to be here, right?
                        if len(beam) < self.beam_width:
                            # We're modifying a list while iterating over it, but since we're adding,
                            # this should be ok.
                            beam.append((new_score, new_split))
                            beam_changed = True
                        else:
                            worst = min(beam)
                            if new_score > worst[0]: # Element 0 of the tuple is the score
                                beam[beam.index(worst)] = (new_score, new_split)
                                beam_changed = True
        
        # TODO: literal pruning for test (see pages 57-58)

        return max(beam)[1] # Element 1 of the tuple is the split
    
    def tests_sig_diff(self, split, new_split):
        raise NotImplementedError

    def fit(self, data, oracle, feature_spec: Tuple[FeatureSpec, ...], max_tree_size = None):
        # Note: parameters passed to fit should represent problem-specific details
        # Max tree size can be seen as problem-specific, so we include it here too.
        if max_tree_size is not None:
            self.max_tree_size = max_tree_size

        self.data = data
        _, self._d = np.shape(data)

        # TODO: Automatic inference of feature spec
        self.feature_spec = feature_spec

        self.oracle = oracle

        # Targets of training data
        targets = self.oracle(data)

        # n is the number of samples
        n: int = data.shape[0]

        # Init root node and tree
        root = Trepan.Node()
        self.tree = root.plant_tree()

        # Root uses all training data
        root.training_idx = np.array(range(n))

        # Root node extra data generator
        root.generator = self.new_generator(data)
        root.generator_training_idx = root.training_idx

        # Root node extra data
        root.gen_data = self.draw_sample((), self.min_sample-n, root.generator)

        # Classify extra data
        root.gen_targets = self.oracle(root.gen_data)

        # Estimate root's prediction
        root.prediction = mode(np.append(targets, root.gen_targets))

        # Estimate root's fidelity
        root.fidelity = np.mean(np.append(targets, root.gen_targets) == root.prediction)

        # Other housekeeping
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
                child.local_constraint = constraint
                child.constraints = node.constraints + (constraint,)

                # Filter training data that gets to the child node
                child.training_idx = [i for i in node.training_idx if constraint.test(data[i])]

                # Re-estimate generator at child node
                if self.same_distribution(child.training_idx, node.generator.training_idx):
                    child.generator = node.generator
                else:
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
        return [self.draw_instance(constraints, generator) for _ in range(max(0,n))]

    def draw_instance(self, constraints, generator, max_attempts = 100):
        
        for _ in range(max_attempts):
            instance = generator()
            if all([c.test(instance) for c in constraints]):
                return instance
        
        raise RuntimeError('Could not generate an acceptable sample within a reasonable time.')
        # TODO: verify against page 50 of thesis
    
    def predict(self, data_matrix):
        return(np.apply_along_axis(self._predict1, 1, data_matrix))

    def _predict1(self, data_vector):

        node = self.tree.root

        while not node.is_leaf:
            for child in node:
                if child.local_constraint.test(data_vector):
                    node = child
                    break
        
        return(node.prediction)
    
    def show_tree(self):

        if self.tree is None:
            "Uninitialized Trepan model"

        return(tree_to_str(self.tree))