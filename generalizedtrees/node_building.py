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

from generalizedtrees.base import SplitTest, GreedyTreeBuilder, null_split
from generalizedtrees.features import FeatureSpec
from generalizedtrees.constraints import Constraint

logger = getLogger()

# For supervised classifiers

class SupervisedClassifierNode:

    def __init__(self, data, targets, target_classes):
        """
        Supervised classifier node constructor

        Data: must be pandas dataframe-like
        Targets: should be pandas series-like.
        """
        super().__init__()
        self.src_data = data
        self.src_targets = targets
        self.target_classes = target_classes
        self.branch = None
        self.idx = None
        self.split: Optional[SplitTest] = None
    
    @property
    def data(self):
        return self.src_data.iloc[self.idx]
    
    @property
    def targets(self):
        return self.src_targets.iloc[self.idx]

    @cached_property
    def probabilities(self):
        slots = pd.Series(0, index=self.target_classes, dtype=np.float)
        freqs = pd.Series(self.targets).value_counts(normalize=True, sort=False)
        return slots.add(freqs, fill_value=0.0)

    def node_proba(self, data_matrix):
        n = data_matrix.shape[0]

        return pd.DataFrame([self.probabilities] * n)
    
    def __str__(self):
        if self.split is None or self.split == null_split:
            return f'Predict {dict(self.probabilities)}'
        else:
            return str(self.split)

class SupCferNodeBuilderMixin:

    def create_root(self):
        node = SupervisedClassifierNode(self.data, self.targets, self.target_classes)
        node.idx = np.arange(node.src_data.shape[0], dtype=np.intp)
        return node

    def generate_children(self, parent):
        # Note: this can be implemented without referencing tree_model or split.
        # Is that always the case?

        # Get branching for training samples
        branches = parent.split.pick_branches(parent.data)

        for b in np.unique(branches):
            node = SupervisedClassifierNode(self.data, self.targets, self.target_classes)
            node.idx = parent.idx[branches == b]
            node.branch = parent.split.constraints[b]

            logger.debug(f'Created node with subview {node.idx}')
            yield node


# For Oracle-and-Generator classifiers

@total_ordering
@dataclass(init=True, repr=True, eq=False, order=False)
class OGClassifierNode:

    training_data: pd.DataFrame
    training_target_proba: pd.DataFrame
    local_constraint: Optional[Constraint]
    constraints: Tuple[Constraint]
    gen_data: pd.DataFrame
    gen_target_proba: pd.DataFrame
    coverage: float
    generator: Any
    split: Optional[SplitTest] = None

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
        if self.split is None or self.split == null_split:
            return f'Predict {dict(self.probabilities)}'
        else:
            return str(self.split)

    @cached_property
    def probabilities(self):
        return self.target_proba.mean(axis=0)

    @cached_property
    def data(self):
        return pd.concat([self.training_data, self.gen_data])
    
    @cached_property
    def target_proba(self):
        return pd.concat([self.training_target_proba, self.gen_target_proba])

    def node_proba(self, data_matrix):
        n = data_matrix.shape[0]

        return pd.DataFrame([self.probabilities] * n)
    
    @cached_property
    def score(self):
        return -(self.coverage * (1 - self.fidelity))
    
    @cached_property
    def fidelity(self):
        # Calculating fidelity as the target probability estimate dotted with itself since
        # The estimate is also the mean of the sample target probabilities
        return sum(self.probabilities * self.probabilities)
    
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


def safe_ask_oracle(data_frame, oracle, result_columns):

    if data_frame.shape[0] < 1:
        return pd.DataFrame(columns=result_columns)
        
    else:
        result = oracle(data_frame)
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result, columns=result_columns)
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

    def create_root(self):
        target_proba = self.oracle(self.data)

        if not isinstance(target_proba, pd.DataFrame):
            target_proba = pd.DataFrame(target_proba)
        
        constraints = ()
        generator = self.new_generator(self.data)

        gen_data = draw_sample(
            constraints,
            self.min_samples - len(self.data),
            self.data.columns,
            generator)
        
        return OGClassifierNode(
            training_data=self.data,
            training_target_proba=target_proba,
            local_constraint=None,
            constraints=constraints,
            gen_data=gen_data,
            gen_target_proba=safe_ask_oracle(gen_data, self.oracle, target_proba.columns),
            coverage=1.0,
            generator=generator)

    def generate_children(self, parent: OGClassifierNode):

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

                    generator = parent.generator

                else:
                    generator = self.new_generator(training_data)

                # Generate data
                gen_data = draw_sample(
                    constraints,
                    self.min_samples - len(training_data),
                    self.data.columns,
                    generator)

                yield OGClassifierNode(
                    training_data = training_data,
                    training_target_proba = parent.training_target_proba[training_mask],
                    local_constraint = local_constraint,
                    constraints = constraints,
                    gen_data = gen_data,
                    gen_target_proba = safe_ask_oracle(
                        gen_data,
                        self.oracle,
                        parent.training_target_proba.columns),
                    coverage = coverage,
                    generator = generator)


# Utility functions (TODO: Find a better-named home)

def draw_sample(constraints, n, columns, generator):
    # Draws samples one at a time.
    # May be worth optimizing.

    logger.debug(f'Drawing sample of size {n}')

    if n < 1:
        return pd.DataFrame(columns=columns)
    else:
        return pd.DataFrame([draw_instance(constraints, generator) for _ in range(n)])

def draw_instance(constraints, generator, max_attempts=1000):

    for _ in range(max_attempts):
        instance = generator.generate()
        if all([c.test(instance) for c in constraints]):
            return instance
    
    logger.critical('Could not generate an acceptable sample within a reasonable time.')
    logger.debug(f'Failed to generate a sample for constraints: {constraints}')
    raise RuntimeError('Could not generate an acceptable sample within a reasonable time.')

def same_distribution(data_1, data_2, feature_spec, alpha):
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
            v1, c1 = np.unique(data_1.iloc[:, i], return_counts=True)
            map1 = {v1[j]: c1[j] for j in range(len(v1))}

            v2, c2 = np.unique(data_2.iloc[:, i], return_counts=True)
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
            _, p = ks_2samp(data_1.iloc[:, i], data_2.iloc[:, i])
            if p < min_p:
                min_p = p
            n_tests += 1

    return min_p < alpha/n_tests