# Split generation components
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from abc import abstractmethod
from logging import getLogger
from typing import Collection, Container, Iterable, Protocol, Optional
from functools import cached_property
from collections import defaultdict
from operator import itemgetter
import heapq

import numpy as np
from scipy.stats import chi2_contingency

from generalizedtrees.constraints import Constraint, MofN, SimpleConstraint, Op, LEQConstraint, GTConstraint, NEQConstraint, EQConstraint
from generalizedtrees.features import FeatureSpec
from generalizedtrees.givens import GivensLC
from generalizedtrees import scores
from generalizedtrees.util import ScoredItem

logger = getLogger()

#################
# Split classes #
#################

# Interface definition
class SplitTest(Protocol):
    """
    Base abstract class for splits.

    A split is defined as a test with integer outcomes and a set of constraints each of which
    corresponds to an outcome of the test.
    """

    @abstractmethod
    def pick_branches(self, data_matrix: np.ndarray):
        raise NotImplementedError

    @property
    @abstractmethod
    def constraints(self) -> Collection[Constraint]:
        raise NotImplementedError


# Implementations

class SplitGT(SplitTest):

    def __init__(self, feature, value, feature_name=None):
        self.feature_name = feature_name
        self.feature = feature
        self.value = value
    
    def pick_branches(self, data_matrix):
        v = data_matrix[:, self.feature]
        try:
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
        return f'Test {feature} > {self.value:.5g}'


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


class BinarySplit(SplitTest):

    def __init__(self, constraint):
        self.constraint = constraint

    @cached_property
    def constraints(self):
        return (~self.constraint, self.constraint)
    
    def pick_branches(self, data_matrix: np.ndarray):
        return self.constraint.test_matrix(data_matrix).astype(np.intp)
    
    def __str__(self):
        return f'Test {str(self.constraint)}'
    
    def __bool__(self):
        # Calling bool on a constraint object should return false if that constraint is vacuous
        return bool(self.constraint)

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
# Split Candidate Generators # - To be deprecated
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


##############################################
# Constraint candidates generating functions #
##############################################

def generate_fayyad_thresholds(
    feature_vector: np.ndarray,
    feature_index: int,
    target_matrix: np.ndarray,
    one_sided: bool
) -> Iterable[Constraint]:
    """
    Generator of split constraints for numeric (or more generally orderable) values.

    We only generate splits between distinctly labeled distinct adjacent values.
    Fayyad and Irani (1992) prove that such splits are always better (in terms of entropy) than
    splits between adjacent equally labeled values.

    :param feature_vectpr: Input data feature vector, length n
    :param feature_index: Index of splitting feature (needed to create split object)
    :param target_matrix: Target value matrix, n-by-k
    :param one_sided: Whether to generate the constraint only on one side of each threshold
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
                yield SimpleConstraint(feature_index, Op.LEQ, split_point)
                if not one_sided:
                    yield SimpleConstraint(feature_index, Op.GT, split_point)
            # Reset collision flag when advancing in x-space
            x_collision = False
        else:
            # Detect y-collision
            if np.any(y_prev != y):
                x_collision = True


def generate_eq_constraints(feature_vector: np.ndarray, feature_index: int, all_but_one: bool) -> Iterable[Constraint]:
    """
    Generator of split constraints for discrete features.
    
    Generates an equality constraint for all (or all but one) of the values in the data.
    :param feature_vector: Feature vector from data
    :param feature_index: Index of splitting feature
    :param all_but_one: Whether to yield all but one (as opposed to all) of the values
    """

    values = np.unique(feature_vector)
    if all_but_one and len(values) > 1: values = values[:-1]

    for x in values:
        yield SimpleConstraint(feature_index, Op.EQ, x)


def generate_atomic_constraints(
    data: np.ndarray,
    y: np.ndarray,
    feature_spec: Container[FeatureSpec],
    one_sided: bool
) -> Iterable[Constraint]:

    for j, spec in enumerate(feature_spec):
        if spec & FeatureSpec.ORDERED:
            yield from generate_fayyad_thresholds(data[:,j], j, y, one_sided)
        elif spec & FeatureSpec.DISCRETE:
            yield from generate_eq_constraints(data[:,j], j, one_sided)
        else:
            raise ValueError(f"I don't know how to handle feature spec {spec}")


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
        n = len(y)
        return scores.entropy_of_label_column(y) - sum(map(
            lambda b: sum(branches == b) / n * scores.entropy_of_label_column(y[branches == b]),
            np.unique(branches)))


class ProbabilityImpurityLC(SplitScoreLC):

    def __init__(self, impurity: str = 'gini') -> None:
        if impurity == 'gini':
            self.impurity = scores.gini_of_p_matrix
            self.weighted_avg = False
        else: #information gain
            self.impurity = scores.entropy_of_p_matrix
            self.weighted_avg = True

    def score(self, node, split: SplitTest, data: np.ndarray, y: np.ndarray) -> float:
        
        branches = split.pick_branches(data)

        if self.weighted_avg:
            n = len(y)
            branch_impurity = lambda b: (branches == b).sum() / n * self.impurity(y[branches == b, :])
        else:
            branch_impurity = lambda b: self.impurity(y[branches == b, :])

        return self.impurity(y) - sum(map(branch_impurity, np.unique(branches)))


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

# Interface definition:

class SplitConstructorLC:

    split_generator: SplitCandidateGeneratorLC
    split_scorer: SplitScoreLC
    only_use_training_to_generate: bool
    only_use_training_to_score: bool
    infimum_score_to_split: float

    def initialize(self, givens: GivensLC) -> None:
        pass

    def construct_split(self, node, data: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> SplitTest:
        raise NotImplementedError()

# Implementations:

class DefaultSplitConstructorLC(SplitConstructorLC):

    def __init__(
        self,
        only_use_training_to_generate: bool = False,
        only_use_training_to_score: bool = False,
        infimum_score_to_split: float = 0
    ) -> None:
        self.only_use_training_to_generate = only_use_training_to_generate
        self.only_use_training_to_score = only_use_training_to_score
        self.infimum_score_to_split = infimum_score_to_split

    def initialize(self, givens: GivensLC) -> None:
        self.split_generator.initialize(givens)

    def construct_split(self, node, data: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> SplitTest:

        if data is None:
            data = node.data
        
        if y is None:
            y = node.y
        
        if self.only_use_training_to_generate:
            g_data = data[:node.n_training, :]
            g_y = y[:node.n_training, :]
        else:
            g_data = data
            g_y = y
        
        if self.only_use_training_to_score:
            s_data = data[:node.n_training, :]
            s_y = y[:node.n_training, :]
        else:
            s_data = data
            s_y = y

        best_split = None
        best_split_score = self.infimum_score_to_split
        for split in self.split_generator.genenerator(g_data, g_y):
            new_score = self.split_scorer.score(node, split, s_data, s_y)
            if new_score > best_split_score:
                best_split_score = new_score
                best_split = split

        return best_split


class MofNSplitConstructorLC(SplitConstructorLC):

    beam_width: int
    alpha: float

    def __init__(
        self,
        beam_width = 2,
        alpha = 0.05,
        only_use_training_to_generate: bool = False,
        only_use_training_to_score: bool = False,
        infimum_score_to_split: float = 0
    ) -> None:
        self.beam_width = beam_width
        self.alpha = alpha
        self.only_use_training_to_generate = only_use_training_to_generate
        self.only_use_training_to_score = only_use_training_to_score
        self.infimum_score_to_split = infimum_score_to_split
    
    def initialize(self, givens: GivensLC) -> None:
        self.split_generator.initialize(givens)

    def construct_split(self, node, data: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> SplitTest:
        if data is None:
            data = node.data
        
        if y is None:
            y = node.y
        
        if self.only_use_training_to_generate:
            g_data = data[:node.n_training, :]
            g_y = y[:node.n_training, :]
        else:
            g_data = data
            g_y = y
        
        if self.only_use_training_to_score:
            s_data = data[:node.n_training, :]
            s_y = y[:node.n_training, :]
        else:
            s_data = data
            s_y = y

        candidate_splits = list(self.split_generator.genenerator(g_data, g_y))

        if not candidate_splits:
            # Unlikely that code gets here but if it does
            return None

        # Find starting point (best split)
        best_split = None
        best_split_score = -np.inf
        for split in candidate_splits:
            new_score = self.split_scorer.score(node, split, s_data, s_y)
            if best_split is None or new_score > best_split_score:
                best_split_score = new_score
                best_split = split
        
        constraint_candidates = [constraint for split in candidate_splits for constraint in split.constraints]

        # Initialize beam
        prev_beam = []

        # M-of-N beam search assumes binary splits but an n-way split could have possibly been returned,
        # in which case the scores for binary splits corresponding to each output constraint would be different
        # from the n-way split score.
        if len(best_split.constraints) > 2:
            beam = []
            for constraint in best_split.constraints:
                heapq.heappush(
                    beam,
                    ScoredItem(
                        score = self.split_scorer.score(node, BinarySplit(constraint), s_data, s_y),
                        item = constraint))
                while len(beam) > self.beam_width:
                    heapq.heappop(beam)
        else:
            beam = [
                ScoredItem(score = best_split_score, item = constraint)
                for constraint in best_split.constraints]

        # Beam search
        while beam != prev_beam:
            logger.debug(f'm-of-n beam search beam: {beam}')
            prev_beam = beam.copy()

            # Trick to iterate over a past snapshot of the beam while modifying the real thing
            for scored_constraint in prev_beam:
                for new_constraint in MofN.neighboring_tests(scored_constraint.item, constraint_candidates):
                    if self.tests_sig_diff(scored_constraint.item, new_constraint, s_data, s_y):
                        new_score = self.split_scorer.score(node, BinarySplit(new_constraint), s_data, s_y)

                        new_scored_constraint = ScoredItem(score = new_score, item = new_constraint)

                        if len(beam) < self.beam_width:
                            heapq.heappush(beam, new_scored_constraint)
                        else:
                            heapq.heappushpop(beam, new_scored_constraint)
            
        # TODO: literal pruning (see pages 57-58)

        return BinarySplit(max(beam).item)
    

    def tests_sig_diff(self, constraint_a, constraint_b, x, y):

        # TODO: check if this test should use y-values, and if so, how?
        #hard_y = y.argmax(axis=0)
        a_branch = np.apply_along_axis(constraint_a.test, axis=1, arr=x)
        b_branch = np.apply_along_axis(constraint_b.test, axis=1, arr=x)

        freq = np.array(
            [[sum(a_branch), sum(~a_branch)],
            [sum(b_branch), sum(~b_branch)]]) + 1 # Smoothing for 0s
        
        _, p, _, _ = chi2_contingency(observed=freq)

        return p < self.alpha
        

class GroupSplitConstructorLC(SplitConstructorLC):

    SUPPORTED_SEARCH_MODES = ('m_of_n', 'groups')

    def __init__(
        self,
        beam_width = 2,
        alpha = 0.05,
        search_mode = 'm_of_n'
    ) -> None:

        self.beam_width: int = beam_width
        self.alpha: float = alpha

        # Validate search mode
        if search_mode in GroupSplitConstructorLC.SUPPORTED_SEARCH_MODES:
            self.search_mode = search_mode
        else:
            raise ValueError(
                f'Unsupported search mode "{search_mode}", '
                f'supported modes are: {GroupSplitConstructorLC.SUPPORTED_SEARCH_MODES}.')

        # Initialized in initialize
        self.feature_groups = None
        self.feature_spec = None

    def initialize(self, givens: GivensLC) -> None:
        # Being explicit about what we're using from givens
        self.feature_spec = givens.feature_spec
        self.feature_groups = givens.feature_groups

        # Verify that all features in the groups are in the spec
        n = len(self.feature_spec)
        self.feature_groups = []
        for group in givens.feature_groups:
            featureSet = set()
            for f in group:
                if f < n:
                    featureSet.add(f)
                else:
                    logger.warning(
                        f'Feature {f} (referenced in feature groups) '
                        'is not in our feature spec!')
            if featureSet:
                self.feature_groups.append(featureSet)
            else:
                logger.warning('Dropping empty feature group!')
    
    def construct_split(self, node, data: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> SplitTest:
        
        if data is None:
            data = node.data
        
        if y is None:
            y = node.y
        
        if self.only_use_training_to_generate:
            g_data = data[:node.n_training, :]
            g_y = y[:node.n_training, :]
        else:
            g_data = data
            g_y = y
        
        if self.only_use_training_to_score:
            s_data = data[:node.n_training, :]
            s_y = y[:node.n_training, :]
        else:
            s_data = data
            s_y = y
        
        all_constraint_candidates = list(
            generate_atomic_constraints(g_data, g_y, feature_spec=self.feature_spec, one_sided = True))

        if not all_constraint_candidates:
            logger.error('There were no constraint candidates for building a split!')
            return None # Or should this be an exception?

        # Will record best split
        best_split = None
        best_split_score = 0

        # For each feature group
        for feature_group in self.feature_groups:
            
            # See if this beats best group-level winner
            winner = self._group_constraints_search(node, s_data, s_y, all_constraint_candidates, feature_group)
            if winner.score > best_split_score:
                best_split_score = winner.score
                best_split = BinarySplit(winner.item)

        return best_split
    
    def _group_constraints_search(self, node, s_data, s_y, all_constraint_candidates, feature_group) -> ScoredItem:

        if self.search_mode == 'm_of_n':
            return self._m_of_n_split_search(node, s_data, s_y, all_constraint_candidates, feature_group)
        if self.search_mode == 'groups':
            return self._groups_split_search(node, s_data, s_y, all_constraint_candidates, feature_group)

        raise ValueError(f'Invalid search mode "{self.search_mode}"')        

    def _groups_split_search(self, node, s_data, s_y, all_constraint_candidates, feature_group) -> ScoredItem:

        # Get best atomic split for each feature in the group
        try:
            best_constraint_scores = {}
            for constraint in all_constraint_candidates:
                f = constraint.feature
                if f in feature_group:
                    score = self.split_scorer.score(node, BinarySplit(constraint), s_data, s_y)
                    if f not in best_constraint_scores or best_constraint_scores[f].score < score:
                        best_constraint_scores[f] = ScoredItem(
                            score=score,
                            item=constraint)

            starting_constraint_dict = {feat: si.item for feat, si in best_constraint_scores.items()}

        except:
            logger.debug('Failure in group split search')
            logger.debug(f'Feature group: {feature_group}')
            logger.debug(f'Constraint candidates: {all_constraint_candidates}')
            raise

        best = ScoredItem(score = 0, item = None)

        def score_cd(m, constraint_dict):
            constraint = MofN(m, list(constraint_dict.values()))
            return ScoredItem(
                score = self.split_scorer.score(
                    node,
                    BinarySplit(constraint),
                    s_data, s_y),
                item = constraint)

        # For each m-value
        for m in range(len(feature_group)+1):

            # Start with constraint thresholds based on best atomic splits
            best_constraint_dict = starting_constraint_dict.copy()

            best_for_m = score_cd(m, best_constraint_dict)
            
            prev_constraint_dict = {}
            # Greedy search 1-feature threshold changes
            while best_constraint_dict != prev_constraint_dict:
                prev_constraint_dict = best_constraint_dict
                for feature in feature_group:
                    for constraint in all_constraint_candidates:
                        if constraint.feature == feature and constraint != best_constraint_dict[feature]:
                            candidate_constraint_dict = best_constraint_dict.copy()
                            candidate = score_cd(m, candidate_constraint_dict)
                            if candidate > best_for_m:
                                best_for_m = candidate
                                best_constraint_dict = candidate_constraint_dict
            
            # Remember best
            if best_for_m > best:
                best = best_for_m

        return best

    def _m_of_n_split_search(self, node, s_data, s_y, all_constraint_candidates, feature_group) -> ScoredItem:

        # Filter constraint candidates to those within the group
        constraint_candidates = [c for c in all_constraint_candidates if c.feature in feature_group]

        # Initialize beam
        prev_beam = []
        beam = []
        for constraint in constraint_candidates:
            new_scored_constraint = ScoredItem(
                    score = self.split_scorer.score(node, BinarySplit(constraint), s_data, s_y),
                    item = constraint)

            if len(beam) < self.beam_width:
                heapq.heappush(beam, new_scored_constraint)
            else:
                heapq.heappushpop(beam, new_scored_constraint)

        # Beam search
        while beam != prev_beam:
            logger.debug(f'm-of-n beam search beam: {beam}')
            prev_beam = beam.copy()

            # Trick to iterate over a past snapshot of the beam while modifying the real thing
            for scored_constraint in prev_beam:
                for new_constraint in MofN.neighboring_tests(scored_constraint.item, constraint_candidates):
                    if self.tests_sig_diff(scored_constraint.item, new_constraint, s_data, s_y):
                        new_score = self.split_scorer.score(node, BinarySplit(new_constraint), s_data, s_y)

                        new_scored_constraint = ScoredItem(score = new_score, item = new_constraint)

                        if len(beam) < self.beam_width:
                            heapq.heappush(beam, new_scored_constraint)
                        else:
                            heapq.heappushpop(beam, new_scored_constraint)
            
        # TODO: literal pruning (see pages 57-58)

        return max(beam)

    def tests_sig_diff(self, constraint_a, constraint_b, x, y):

        # TODO: check if this test should use y-values, and if so, how?
        #hard_y = y.argmax(axis=0)
        a_branch = np.apply_along_axis(constraint_a.test, axis=1, arr=x)
        b_branch = np.apply_along_axis(constraint_b.test, axis=1, arr=x)

        freq = np.array(
            [[sum(a_branch), sum(~a_branch)],
            [sum(b_branch), sum(~b_branch)]]) + 1 # Smoothing for 0s
        
        _, p, _, _ = chi2_contingency(observed=freq)

        return p < self.alpha
