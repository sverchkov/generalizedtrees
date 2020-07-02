# Implementation of Trepan (Craven and Shavlik 1996)
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
from functools import total_ordering
from generalizedtrees import splitting
from generalizedtrees.base.tree_estimator import TreeEstimatorMixin, TreeEstimatorNode
from generalizedtrees.base.tree_learner import TreeLearner
from generalizedtrees.queues import Heap
from generalizedtrees.core import FeatureSpec
from generalizedtrees.constraints import MofN
from generalizedtrees.leaves import SimplePredictor
from generalizedtrees.sampling import rejection_sample_generator
from generalizedtrees.scores import gini, entropy
from generalizedtrees.splitting import fayyad_thresholds, one_vs_all
from generalizedtrees.tree import TreeNode, Tree, tree_to_str
from heapq import heappush, heappop
from scipy.stats import mode, ks_2samp, chisquare
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KernelDensity
from statistics import mode
from typing import Tuple


logger = logging.getLogger()


@total_ordering
class TrepanNode(TreeEstimatorNode):

    def __init__(self):
        super().__init__()
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

    def __eq__(self, other):
        if self.score is None or other.score is None:
            return False
        else:
            return self.score == other.score
    
    def __lt__(self, other):
        if self.score is None or other.score is None:
            raise ValueError("Unable to compare nodes with uninitialized scores")
        else:
            return self.score < other.score

    def __str__(self):

        if self.depth == 0:
            return 'Root'

        elif self.is_leaf:
            return f'If {self.local_constraint} predict {self.prediction}'
        
        else:
            return f'If {self.local_constraint}'
    
    def pick_branch(self, data_vector):
    
        for i in range(len(self)):
            if self[i].local_constraint.test(data_vector):
                return i
        
        logger.error(f"""\
            Unable to pick branch for:
            data vector: {data_vector}
            node: {self}""")
        raise ValueError

    def predict(self, data_vector):
        return self.prediction


class Trepan(TreeLearner, TreeEstimatorMixin):
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
        min_sample = 20,
        dist_test_alpha = 0.05,
        use_m_of_n = True,
        beam_width = 2):
        # Note: parameters passed to init sould define *how* the explanation tree is built
        # Note: init should declare all members

        super().__init__()

        self.data: np.ndarray
        self.targets: np.ndarray
        self.oracle = None # f(x) -> y
        self.train_features = None # training set x (no y's)
        self.feature_spec: Tuple[FeatureSpec, ...] # Tuple of feature specs
        self.min_sample = min_sample
        self.rng = rng
        self.dist_test_alpha = dist_test_alpha
        self.use_m_of_n = use_m_of_n
        self.beam_width = beam_width

        # Inferred values
        self._d: int

        # Stopping criteria:
        self.max_tree_size: int = max_tree_size

    Generator = namedtuple("Generator", ["generate", "training_idx"])

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
            generate = lambda: np.reshape([f() for f in feature_generators], (1, self._d)),
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

    def construct_split(self, node: TrepanNode):
        data = np.append(self.data[node.training_idx, :], node.gen_data, axis=0)
        targets = np.append(self.targets[node.training_idx], node.gen_targets)

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
        """
        Compute the split score (information gain) for a split.
        """
        return entropy(targets) - sum(map(
            lambda c: entropy(targets[np.apply_along_axis(c.test, 1, data)]),
            split))

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
        _, self._d = np.shape(self.data)

        # TODO: Automatic inference of feature spec
        self.feature_spec = feature_spec

        self.oracle = oracle

        # Targets of training data
        self.targets = self.oracle(self.data)

        # Build the tree
        self._build()

        return self

    def draw_sample(self, constraints, n, generator):
        # Original implementation draws samples one at a time.
        # May be worth optimizing.

        logger.debug(f'Drawing sample of size {n}')

        if n < 1:
            return np.zeros((0, self._d)), np.array([])
        else:
            data = np.vstack(
                [self.draw_instance(constraints, generator) for _ in range(n)])
            targets = self.oracle(data)

            logger.debug(f'Produced data shape {data.shape} and targets shape {targets.shape}')

            return data, targets

    def draw_instance(self, constraints, generator, max_attempts = 100):
        
        for _ in range(max_attempts):
            instance = generator.generate()
            if all([c.test(instance.flatten()) for c in constraints]):
                logger.debug(f'produced instance shape {instance.shape}')
                return instance
        
        raise RuntimeError('Could not generate an acceptable sample within a reasonable time.')
        # TODO: verify against page 50 of thesis

    def new_node(self, branch = None, parent = None):

        # For Trepan, branches are just constraint objects.

        node = TrepanNode()

        if branch is not None and parent is not None:

            node.local_constraint = branch
            node.constraints = parent.constraints + (node.local_constraint,)

            # Filter training data that gets to the node
            node.training_idx = [
                i for i in parent.training_idx
                if node.local_constraint.test(self.data[i])]
        
        else:
            node.constraints = ()
            node.training_idx = list(range(self.data.shape[0]))
            
        # Re-estimate generator at node
        if parent is not None and self.same_distribution(node.training_idx, parent.generator.training_idx):
            node.generator = parent.generator
        else:
            node.generator = self.new_generator(node.training_idx)

        # Generate and classify data for node
        node.gen_data, node.gen_targets = self.draw_sample(
            node.constraints,
            self.min_sample-len(node.training_idx),
            node.generator)

        # Compute node's prediction
        node.prediction = mode(np.append(self.targets[node.training_idx], node.gen_targets))

        # Estimate node's fidelity
        node.fidelity = np.mean(np.append(self.targets, node.gen_targets) == node.prediction)

        # Estimate node's reach
        if branch is not None and parent is not None:
            n_gen = parent.gen_data.shape[0]
            if n_gen > 0:
                n_accepted = sum(np.apply_along_axis(node.local_constraint.test, 1, parent.gen_data))
            else:
                n_accepted = 0

            node.coverage = \
                (len(node.training_idx) + n_accepted) / \
                (len(parent.training_idx) + n_gen) * \
                parent.coverage
        
        else:
            node.coverage = 1

        # Node score is the negative of:
        #   f(N) = reach(N) (1-fidelity(N))
        node.score = -(node.coverage * (1 - node.fidelity))

        return node

    def new_queue(self):
        return Heap()

    def global_stop(self):
        return self.tree.size >= self.max_tree_size

    def local_stop(self, node):
        return all( self.targets[node.training_idx] == node.prediction)

