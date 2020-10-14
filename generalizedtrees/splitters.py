# Splitter functions used in tree-building
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
from logging import getLogger
from generalizedtrees.scores import entropy, entropy_of_p_matrix
from generalizedtrees.splitting import fayyad_thresholds, fayyad_thresholds_p, one_vs_all
from generalizedtrees.features import FeatureSpec

logger = getLogger()


def make_split_candidates(feature_spec, data, targets):
    # Note: we could make splits based on original training examples only, or on
    # training examples and generated examples.
    # In the current form, this could be controlled by the calling function.

    result = []

    for j in range(len(feature_spec)):
        if feature_spec[j] is FeatureSpec.CONTINUOUS:
            result.extend(fayyad_thresholds(data, targets, j))
        elif feature_spec[j] & FeatureSpec.DISCRETE:
            result.extend(one_vs_all(data, j))
        else:
            raise ValueError(f"I don't know how to handle feature spec {feature_spec[j]}")
    
    return result


def make_split_candidates_p(feature_spec, data, target_proba):

    result = []

    for j in range(len(feature_spec)):
        if feature_spec[j] is FeatureSpec.CONTINUOUS:
            result.extend(fayyad_thresholds_p(data, target_proba, j))
        elif feature_spec[j] & FeatureSpec.DISCRETE:
            result.extend(one_vs_all(data, j))
        else:
            raise ValueError(f"I don't know how to handle feature spec {feature_spec[j]}")
    
    return result


# Score functions for splits

def information_gain(split, data, targets):
    """
    Compute the split score (information gain) for a split.

    A split is a tuple of constraints.
    """
    branches = split.pick_branches(data)
    return entropy(targets) - sum(map(
        lambda b: entropy(targets[branches == b]),
        np.unique(branches)))

def information_gain_p(split, data, target_proba):
    """
    Compute the split score (information gain) for a split.

    A split is a tuple of constraints.

    This form uses probability estimates for targets.
    """
    branches = split.pick_branches(data)

    return entropy_of_p_matrix(target_proba) - sum(map(
        lambda b: entropy_of_p_matrix(target_proba[branches==b, :]),
        np.unique(branches)
    ))


def ijcai19_lr_gradient_slow(node, split):
    """
    Gradient-based split score for logistic regression model trees.

    A slow implementation of the criterion described in Broelemann and Kasneci 2019 (IJCAI)
    Assumes that the model at the node is an sk-learn binary logistic regression.
    """

    branches = split.pick_branches(node.data)

    x = np.concatenate([np.ones((node.data.shape[0], 1)), node.data], axis=1)

    # LR loss (unregularized) gradient is easy to compute from data, targets, and prediction:
    y = node.target_proba[:,[1]]
    y_hat = node.model.estimate(node.data)[:,[1]]
    gradients = (y_hat - y) * y_hat * (1 - y_hat) * x

    # This is eq. 6 in the paper
    ssg = (gradients**2).sum(axis=1)
    return sum(map(lambda b: ssg[branches == b].mean(), np.unique(branches)))


def auroc_criterion(split, data, target_proba):
    """
    Compute a criterion based on area under an ROC curve.

    Adapted from the AUCsplit criterion (Ferri et al. ICML 2002) for target probability matrices.
    PDF at: http://staff.icar.cnr.it/manco/Teaching/2006/datamining/articoli/105.pdf
    """

    n_s, n_t = target_proba.shape

    thresholds = np.unique(target_proba[:, 1], return_index = True)
    
    if n_t != 2:
        raise ValueError("The auroc criterion is only available for binary classification")

    k = len(split.constraints)

    branches = split.pick_branches(data)

    best_auroc = 0

    for t in thresholds:
        positives = target_proba[:,1] >= t
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


