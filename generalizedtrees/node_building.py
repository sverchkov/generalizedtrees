# Mixins for node-building in tree learners and supporting objects
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

from typing import Optional, Tuple, Callable, Any
from functools import cached_property, total_ordering
from dataclasses import dataclass

import numpy as np
import pandas as pd
from logging import getLogger
from scipy.stats import ks_2samp, chisquare
from sklearn.linear_model import LogisticRegression

from generalizedtrees.base import order_by, SplitTest, GreedyTreeBuilder
from generalizedtrees.features import FeatureSpec
from generalizedtrees.constraints import Constraint
from generalizedtrees.leaves import ConstantEstimator, SKProbaEstimator

logger = getLogger()


@dataclass
class SupervisedClassifierNode:
    data: np.ndarray
    targets: np.ndarray
    split: Optional[SplitTest] = None
    local_constraint: Optional[Constraint] = None
    model: Optional[ConstantEstimator] = None


class SupCferNodeBuilderMixin:

    def create_root(self):
        node = SupervisedClassifierNode(
            data = self.data,
            targets = self.targets,
            model = ConstantEstimator())
        
        node.model.fit(self.targets)
        return node

    def generate_children(self, parent):

        # Get branching for training samples
        branches = parent.split.pick_branches(parent.data)

        for b in np.unique(branches):
            idx = branches == b
            node = SupervisedClassifierNode(
                data = parent.data[idx, :],
                targets = parent.targets[idx, :],
                local_constraint = parent.split.constraints[b],
                model = ConstantEstimator())
            
            node.model.fit(node.targets)

            yield node


# Builder for trepan

@order_by("score")
@dataclass(init=True, repr=True, eq=False, order=False)
class TrepanNode:

    training_data: np.ndarray
    training_target_proba: np.ndarray
    local_constraint: Optional[Constraint]
    constraints: Tuple[Constraint]
    gen_data: np.ndarray
    gen_target_proba: np.ndarray
    coverage: float
    generate: Callable
    split: Optional[SplitTest] = None
    
    @cached_property
    def data(self):
        return np.concatenate([self.training_data, self.gen_data], axis=0)
    
    @cached_property
    def target_proba(self):
        return np.concatenate([self.training_target_proba, self.gen_target_proba], axis=0)
    
    @cached_property
    def score(self):
        return -(self.coverage * (1 - self.fidelity))
    
    @cached_property
    def fidelity(self):
        return sum((self.target_proba * self.model.estimate(self.data)).mean(axis=0))

    @cached_property
    def model(self) -> ConstantEstimator:
        return ConstantEstimator().fit(self.target_proba)


def safe_ask_oracle(data_matrix, oracle, d):

    if data_matrix.shape[0] < 1:
        return np.empty((0, d))
        
    else:
        result = oracle(data_matrix)
        if not isinstance(result, np.ndarray):
            result = result.to_numpy()
        return result


class OGCferNodeBuilderMixin:
    # Contract to formalize:
    # Requires parameters:
    # - data
    # - oracle
    # - new_generator
    # - same_distribution
    # - min_samples
    # - feature_spec
    # - dist_test_alpha
    # - node_cls (optional, defults to TrepanNode)

    def create_root(self):

        Node = getattr(self, "node_cls", TrepanNode)

        target_proba = self.oracle(self.data)

        if not isinstance(target_proba, pd.DataFrame):
            target_proba = pd.DataFrame(target_proba)
        
        constraints = ()
        generate = self.new_generator(self.data)

        gen_data = draw_sample(
            constraints,
            self.min_samples - len(self.data),
            self._d,
            generate)
        
        return Node(
            training_data=self.data,
            training_target_proba=target_proba,
            local_constraint=None,
            constraints=constraints,
            gen_data=gen_data,
            gen_target_proba=safe_ask_oracle(gen_data, self.oracle, target_proba.shape[1]),
            coverage=1.0,
            generate=generate)

    def generate_children(self, parent):

        # Design decision: 
        # Infer child node classes from parent class. Seems most flexible.
        Node = type(parent)
        
        if parent.split is None: return

        branches = parent.split.pick_branches(parent.training_data)
        gen_branches = parent.split.pick_branches(parent.gen_data)

        unique_branches = np.unique(branches)

        if len(unique_branches) > 1:

            for b in unique_branches:

                training_mask = branches == b
                local_constraint = parent.split.constraints[b]
                constraints = parent.constraints + (local_constraint,)
                training_data = parent.training_data[training_mask]

                # Compute coverage
                coverage = \
                    (sum(training_mask) + sum(gen_branches == b)) / \
                    (len(training_mask) + len(gen_branches)) * \
                    parent.coverage

                # Make generator                
                if same_distribution(
                    training_data,
                    parent.training_data,
                    self.feature_spec,
                    self.dist_test_alpha):

                    generate = parent.generate

                else:
                    generate = self.new_generator(training_data)

                # Generate data
                gen_data = draw_sample(
                    constraints,
                    self.min_samples - len(training_data),
                    self._d,
                    generate)

                yield Node(
                    training_data = training_data,
                    training_target_proba = parent.training_target_proba[training_mask],
                    local_constraint = local_constraint,
                    constraints = constraints,
                    gen_data = gen_data,
                    gen_target_proba = safe_ask_oracle(
                        gen_data,
                        self.oracle,
                        parent.training_target_proba.shape[1]),
                    coverage = coverage,
                    generate = generate)


# Utility functions (TODO: Find a better-named home)

def draw_sample(constraints, n, d, generator):
    # Draws samples one at a time.
    # May be worth optimizing.

    logger.debug(f'Drawing sample of size {n}')

    if n < 1:
        return np.zeros((0, d))
    else:
        return np.array([draw_instance(constraints, generator) for _ in range(n)])

def draw_instance(constraints, generate, max_attempts=1000):

    for _ in range(max_attempts):
        instance = generate()
        if all([c.test(instance) for c in constraints]):
            return instance
    
    logger.critical('Could not generate an acceptable sample within a reasonable time.')
    logger.debug(f'Failed to generate a sample for constraints: {constraints}')
    raise RuntimeError('Could not generate an acceptable sample within a reasonable time.')

def same_distribution(data_1, data_2, /, feature_spec, alpha):
    """
    Performs statistical test to determine if data are from the
    same distribution.

    data_1 and data_2 need to be pandas dataframes
    """
    n_tests = 0
    min_p = 1.0
    for i in range(len(feature_spec)):

        if feature_spec[i] & FeatureSpec.DISCRETE:
            # Get frequencies.
            # Note: we're assuming that the union of values present in the samples is
            # the set of possible values. This is not all possible values that the
            # variable could originally take.
            v1, c1 = np.unique(data_1[:, i], return_counts=True)
            map1 = {v1[j]: c1[j] for j in range(len(v1))}

            v2, c2 = np.unique(data_2[:, i], return_counts=True)
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
            _, p = ks_2samp(data_1[:, i], data_2[:, i])
            if p < min_p:
                min_p = p
            n_tests += 1

    return min_p < alpha/n_tests


@dataclass(init=True, repr=True, eq=False, order=False)
class BATNode:
    data: np.ndarray
    target_proba: np.ndarray
    training_idx: np.ndarray
    gen_idx: np.ndarray
    local_constraint: Optional[Constraint]
    constraints: Tuple[Constraint]
    cost: float
    model: ConstantEstimator
    split: Optional[SplitTest] = None


class BATNodeBuilderMixin:
    """
    Node builder following the scheme in Born-Again Trees.
    """
    # Contract to formalize:
    # Requires parameters:
    # - data
    # - oracle
    # - new_generator
    # - same_distribution
    # - min_samples
    # - feature_spec
    # - dist_test_alpha

    def _fetch_sample(self, constraints, known_training_idx=None, known_gen_idx=None):
        
        if not hasattr(self, "_generate"):
            self._generate = self.new_generator(self.data)
        
        if known_training_idx is None:
            known_training_idx = np.array([
                i for i in range(self.data.shape[0])
                if all(c.test(self.data[i, :]) for c in constraints)
            ], dtype=np.intp)
        
        if not hasattr(self, "_generated_samples"):

            self._generated_samples = pd.DataFrame()
            known_gen_idx = np.empty((0,), dtype=np.intp)

        else:
            if known_gen_idx is None:
                known_gen_idx = np.array([
                    i for i in range(self._generated_samples)
                    if all(c.test(self._generated_samples[i, :]) for c in constraints)
                ], dtype=np.intp)

        # Generate new samples until min_samples is met for constraints
        generated_samples = []
        accepted_gen_idx = []
        past_n_gen = len(self._generated_samples)
        n_needed = self.min_samples - len(known_training_idx) - len(known_gen_idx)

        while len(accepted_gen_idx) < n_needed:

            new_sample = self._generate()

            if all(c.test(new_sample) for c in constraints):
                accepted_gen_idx.append(past_n_gen + len(generated_samples))
            
            generated_samples.append(new_sample)

        if len(generated_samples) > 0:

            self._generated_samples = self._generated_samples.append(generated_samples, ignore_index=True)
            known_gen_idx = np.concatenate([known_gen_idx, np.array(accepted_gen_idx, dtype=np.intp)])
        
        prop_accepted = (
            (self.data.shape[0] + self._generated_samples.shape[0])
            / (len(known_training_idx) + len(known_gen_idx)))
        
        return (
            self._generated_samples.iloc[known_gen_idx, :],
            known_gen_idx,
            known_training_idx,
            prop_accepted)


    def create_root(self):
        # Inefficiency: We already compute something equivalent at fit time
        self._target_proba = self.oracle(self.data)
        
        constraints = ()

        gen_data, training_idx, gen_idx, prop_accepted = self._fetch_sample(constraints)

        # Precompute probabilities
        target_proba = np.concatenate([
            self._target_proba,
            safe_ask_oracle(gen_data, self.oracle, self._target_proba.shape[1])
            ], axis=0)
        
        model = ConstantEstimator().fit(target_proba)
        probabilities = target_proba.mean(axis=0)

        cost = prop_accepted * (1 - max(probabilities))
        
        data_source = self.data[training_idx, :]
        if len(gen_data) > 1:
            data_source = np.concatenate([data_source, gen_data], axis=0)

        return BATNode(
            data=data_source,
            target_proba=target_proba,
            training_idx=training_idx,
            gen_idx=gen_idx,
            local_constraint=None,
            constraints=constraints,
            model=model,
            cost=cost)

    def generate_children(self, parent):

        if parent.split is None: return

        branches = parent.split.pick_branches(self.data(parent.training_idx))
        gen_branches = parent.split.pick_branches(self._generated_samples(parent.gen_idx))

        unique_branches = np.unique(branches)

        # Born again trees rule is not to split if not all branches are
        # populated by training samples.
        if len(unique_branches) < len(parent.split.constraints): return

        for b in unique_branches:

            training_mask = branches == b
            local_constraint = parent.split.constraints[b]
            constraints = parent.constraints + (local_constraint,)
            training_idx = parent.training_idx[training_mask]

            # Make use of parent generated instances if possible
            gen_idx = parent.gen_idx[gen_branches == b]

            # Generate data
            gen_data, training_idx, gen_idx, prop_accepted = self._fetch_sample(
                constraints,
                training_idx,
                gen_idx)

            # Precompute probabilities
            target_proba = pd.concat([
                self._target_proba[training_idx, :],
                safe_ask_oracle(gen_data, self.oracle, parent._target_proba.shape[1])
                ],ignore_index=True)
            model = ConstantEstimator().fit(target_proba)
            probabilities = target_proba.mean(axis=0)

            cost = prop_accepted * (1 - max(probabilities))

            yield BATNode(
                data=self.data[training_idx, :].append(gen_data, ignore_index=True),
                target_proba=target_proba,
                training_idx=training_idx,
                gen_idx=gen_idx,
                local_constraint=local_constraint,
                constraints=constraints,
                model=model,
                cost=cost)


#########################################
# Builder for trepan with linear leaves #
#########################################


@order_by("score")
@dataclass(init=True, repr=True, eq=False, order=False)
class TLLNode:

    training_data: np.ndarray
    training_target_proba: np.ndarray
    local_constraint: Optional[Constraint]
    constraints: Tuple[Constraint]
    gen_data: np.ndarray
    gen_target_proba: np.ndarray
    coverage: float
    generate: Callable
    split: Optional[SplitTest] = None

    def __str__(self):
        if self.split is None:
            return str(self.model)
        else:
            return str(self.split)
    
    @cached_property
    def model(self) -> SKProbaEstimator:

        # TODO: Expose LR parameters to the outside
        est = SKProbaEstimator(
            LogisticRegression(
                penalty='l1',
                C=0.1,
                multi_class='multinomial',
                solver='saga')
        )

        # Prepare data for fit
        # Use sample weights to communicate probabilities to the learner
        n, m = self.target_proba.shape

        lr_data = np.concatenate([self.data]*m, axis=0)
        lr_sample_weights = self.target_proba.flatten()
        lr_targets = np.repeat(np.arange(m), n)

        return est.fit(lr_data, lr_targets, sample_weight=lr_sample_weights)

    @cached_property
    def classifier(self):
        return self.target_proba.mean(axis=0)

    @cached_property
    def data(self):
        return np.concatenate([self.training_data, self.gen_data], axis=0)
    
    @cached_property
    def target_proba(self) -> np.ndarray:
        return np.concatenate([self.training_target_proba, self.gen_target_proba], axis=0)
    
    @cached_property
    def score(self):
        return -(self.coverage * (1 - self.fidelity))
    
    @cached_property
    def fidelity(self):
        return sum((self.target_proba * self.model.estimate(self.data)).mean(axis=0))

    @cached_property
    def targets(self):
        # For compatibility with split selectors that use targets
        try:
            return self.target_proba.idxmax(axis=1)
        except:
            logger.critical(
                'Something went wrong when inferring hard target classes.'
                'Target probability vector:'
                f'\n{self.target_proba}')
            raise

