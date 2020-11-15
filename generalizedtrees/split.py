# Split generation components
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

from abc import abstractmethod
from logging import getLogger
from typing import Container, Iterable, Protocol, Optional
from functools import cached_property
from operator import itemgetter

import numpy as np

from generalizedtrees.base import SplitTest
from generalizedtrees.constraints import LEQConstraint, GTConstraint, NEQConstraint, EQConstraint
from generalizedtrees.features import FeatureSpec
from generalizedtrees.givens import GivensLC
from generalizedtrees import scores

logger = getLogger()

#################
# Split classes #
#################

class SplitGT(SplitTest):

    def __init__(self, feature, value, feature_name=None):
        self.feature_name = feature_name
        self.feature = feature
        self.value = value
    
    def pick_branches(self, data_matrix):
        try:
            v = data_matrix[:, self.feature]
            return (v > self.value).astype(np.intp)
        except:
            logger.fatal(
                'Something went wrong.\n'
                f'Feature: {self.feature}, Value: {self.value}\n'
                f'Matrix:\n{data_matrix}\n'
                f'Column returned:\n{v}')
            raise

    @cached_property
    def constraints(self):
        return (
            LEQConstraint(self.feature, self.value),
            GTConstraint(self.feature, self.value))
    
    def __str__(self):
        if self.feature_name is None:
            feature = f'x[{self.feature}]'
        else:
            feature = self.feature_name
        return f'Test {feature} > {self.value}'


class SplitOneVsAll(SplitTest):

    def __init__(self, feature, value, feature_name=None):
        self.feature_name = feature_name
        self.feature = feature
        self.value = value
    
    def pick_branches(self, data_matrix):
        return (data_matrix[:, self.feature] == self.value).astype(np.intp)

    @cached_property
    def constraints(self):
        return (
            NEQConstraint(self.feature, self.value),
            EQConstraint(self.feature, self.value))

    def __str__(self):
        if self.feature_name is None:
            feature = f'x[{self.feature}]'
        else:
            feature = self.feature_name
        return f'Test {feature} == {self.value}'


class SplitEveryValue(SplitTest):

    def __init__(self, feature, values, feature_name=None):
        self.feature_name = feature_name
        self.feature = feature
        self.values = values
        self.map = {values[i]: i for i in range(len(values))}
    
    def pick_branches(self, data_matrix):
        return data_matrix[:, self.feature].map(self.map)

    @cached_property
    def constraints(self):
        return (EQConstraint(self.feature, v) for v in self.values)

    def __str__(self):
        if self.feature_name is None:
            feature = f'x[{self.feature}]'
        else:
            feature = self.feature_name
        return f'Test {feature} against each of {self.values}'


###################################
# Base split generating functions #
###################################


def fayyad_thresholds(feature_vector, feature_index, target_matrix):
    """
    Generator of splits for numeric (or more generally orderable) values.

    We only generate splits between distinctly labeled distinct adjacent values.
    Fayyad and Irani (1992) prove that such splits are always better (in terms of entropy) than
    splits between adjacent equally labeled values.

    :param data: Input data feature vector, length n
    :param feature: Index of splitting feature (needed to create split object)
    :param target: Target value matrix, n-by-k
    """
    v = sorted(zip(feature_vector, target_matrix), key=itemgetter(0))
    
    # Flag for handling the case when two identical x-values have distinct y-values
    x_collision = False

    for j in range(1, len(v), 1):
        x_prev, y_prev = v[j-1]
        x, y = v[j]

        # Only place splits between distinct x-values
        if x_prev < x:
            # Only place splits between distinct y-values
            if x_collision or np.any(y_prev != y):
                split_point = (x_prev + x)/2
                yield SplitGT(feature_index, split_point)
            # Reset collision flag when advancing in x-space
            x_collision = False
        else:
            # Detect y-collision
            if np.any(y_prev != y):
                x_collision = True


def one_vs_all(feature_vector, feature_index):
    """
    Generator for one-vs-all splits.
    :param data: Input data matrix, n-by-d
    :param feature: Index of splitting feature (integer >=0 and <d)
    """

    values = np.unique(feature_vector)
    
    for x in values:
        yield SplitOneVsAll(feature_index, x)


def all_values_split(feature, values):
    """
    Generator for all-value splits.
    :param feature: Feature index
    :param values: List of possible feature values
    """
    yield SplitEveryValue(feature, values)


##############################
# Split Candidate Generators #
##############################


# Interface definition
class SplitCandidateGeneratorLC(Protocol):

    @abstractmethod
    def initialize(self, givens: GivensLC) -> 'SplitCandidateGeneratorLC':
        raise NotImplementedError

    @abstractmethod
    def genenerator(self, data: np.ndarray, y: np.ndarray) -> Iterable[SplitTest]:
        raise NotImplementedError


# Default implementation for single-feature axis-aligned splits:
class AxisAlignedSplitGeneratorLC(SplitCandidateGeneratorLC):

    feature_spec: Container[FeatureSpec]

    def initialize(self, givens: GivensLC) -> 'SplitCandidateGeneratorLC':
        self.feature_spec = givens.feature_spec
        return self

    def genenerator(self, data: np.ndarray, y: np.ndarray) -> Iterable[SplitTest]:

        for j in range(len(self.feature_spec)):
            if self.feature_spec[j] is FeatureSpec.CONTINUOUS:
                yield from fayyad_thresholds(data[:, j], j, y)
            elif self.feature_spec[j] & FeatureSpec.DISCRETE:
                yield from one_vs_all(data[:, j], j)
            else:
                raise ValueError(f"I don't know how to handle feature spec {self.feature_spec[j]}")


####################################
# Split Scorers Learning Component #
####################################

# Interface definition:
# We could maybe define this as a callable instead?
class SplitScoreLC(Protocol):

    @abstractmethod
    def score(self, node, split: SplitTest, data: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError


# Implementations

class DiscreteInformationGainLC(SplitScoreLC):

    def score(self, node, split: SplitTest, data: np.ndarray, y: np.ndarray) -> float:
        branches = split.pick_branches(data)
        return scores.entropy_of_label_column(y) - sum(map(
            lambda b: scores.entropy_of_label_column(y[branches == b]),
            np.unique(branches)))


class ProbabilityImpurityLC(SplitScoreLC):

    def __init__(self, impurity: str = 'gini') -> None:
        if impurity == 'gini':
            self.impurity = scores.gini_of_p_matrix
        else: #information gain
            self.impurity = scores.entropy_of_p_matrix

    def score(self, node, split: SplitTest, data: np.ndarray, y: np.ndarray) -> float:
        
        branches = split.pick_branches(data)

        return self.impurity(y) - sum(map(
            lambda b: self.impurity(y[branches==b, :]),
            np.unique(branches)
        ))


class IJCAI19LRGradientScoreLC(SplitScoreLC):
    """
    Gradient-based split score for logistic regression model trees.

    A slow implementation of the criterion described in Broelemann and Kasneci 2019 (IJCAI)
    Assumes that the model at the node is an sk-learn binary logistic regression.
    """

    def score(self, node, split: SplitTest, data: np.ndarray, y: np.ndarray) -> float:

        branches = split.pick_branches(data)

        x = np.concatenate([np.ones((data.shape[0], 1)), data], axis=1)

        # LR loss (unregularized) gradient is easy to compute from data, targets, and prediction:
        y = y[:,[1]]
        node_est = node.model.estimate(data)
        if node_est.shape[1] == 2:
            y_hat = node_est[:,[1]]
        else:
            y_hat = node_est
        gradients = (y_hat - y) * y_hat * (1 - y_hat) * x

        # This is eq. 6 in the paper
        ssg = (gradients**2).sum(axis=1)
        return sum(map(lambda b: ssg[branches == b].mean(), np.unique(branches)))


class AUROCSplitScoreLC(SplitScoreLC):
    """
    Compute a criterion based on area under an ROC curve.

    Adapted from the AUCsplit criterion (Ferri et al. ICML 2002) for target probability matrices.
    PDF at: http://staff.icar.cnr.it/manco/Teaching/2006/datamining/articoli/105.pdf
    """

    def score(self, node, split: SplitTest, data: np.ndarray, y: np.ndarray) -> float:

        n_s, n_t = y.shape

        thresholds = np.unique(y[:, 1], return_index = True)
        
        if n_t != 2:
            raise ValueError("The auroc criterion is only available for binary classification")

        k = len(split.constraints)

        branches = split.pick_branches(data)

        best_auroc = 0

        for t in thresholds:
            positives = y[:,1] >= t
            negatives = np.logical_not(positives)
            
            branches_onehot = np.eye(k)[branches]
            branch_pos = positives * branches_onehot
            branch_neg = negatives * branches_onehot

            branch_pos_count = np.sum(branch_pos, axis=0)
            branch_neg_count = np.sum(branch_neg, axis=0)
            branch_count = np.sum(branches_onehot)

            # "Local predictive accuracy"
            if branch_pos_count == branch_neg_count: # Catches 0-sample branches
                branch_lpa = 0.5
            else:
                branch_lpa = branch_pos_count / (branch_pos_count + branch_neg_count)
            
            # Identify branches that have more negatives than positives, update lpa
            neg_branches = branch_lpa < 0.5
            if any(neg_branches):
                branch_lpa[neg_branches] = 1 - branch_lpa[neg_branches]
                branch_pos_count[neg_branches] = branch_neg_count[neg_branches]
                branch_neg_count[neg_branches] = branch_count[neg_branches] - branch_pos_count[neg_branches]
            
            branch_order = np.argsort(branch_lpa)

            auroc = 0
            # Using notation from the paper:
            x_t = sum(negatives)
            y_t = sum(positives)
            y_im1 = 0

            for i in branch_order:
                auroc += branch_neg_count[i] / x_t * (2 * y_im1 + branch_pos_count[i]) / (2 * y_t)
                y_im1 += branch_pos_count[i]

            if auroc > best_auroc:
                best_auroc = auroc

        return best_auroc


#####################################
# Split Generator Learner Component #
#####################################

# Future: define protocol if we ever need to define multiple ways to do this


class SplitConstructorLC:

    split_generator: SplitCandidateGeneratorLC
    split_scorer: SplitScoreLC

    def construct_split(self, node, data: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> SplitTest:

        if data is None:
            data = node.data
        
        if y is None:
            y = node.y
        
        best_split = None
        best_split_score = 0
        for split in self.split_generator.genenerator(data, y):
            new_score = self.split_scorer.score(node, split, data, y)
            if new_score > best_split_score:
                best_split_score = new_score
                best_split = split

        return best_split

