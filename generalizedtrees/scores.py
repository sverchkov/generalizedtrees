# Scoring functions
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020-2021, Yuriy Sverchkov

import numpy as np


def gini_of_label_column(y):

    n: int = len(y)

    if n == 0:
        return 0

    p = np.unique(y, return_counts=True)[1] / n

    return 1 - np.sum(p*p)


def gini_of_p_matrix(pm: np.ndarray):

    if len(pm) == 0: return 0

    p = pm.mean(axis=0)

    return np.sum(p * (1 - p))


def entropy_of_label_column(y):

    n = len(y)

    if n == 0:
        return 0

    p = np.unique(y, return_counts=True)[1] / n

    return entropy_of_p_vector(p)


def entropy_of_p_matrix(pm):

    return entropy_of_p_vector(np.mean(pm, axis=0))


def entropy_of_p_vector(p):
    
    pl2p = np.where(p > 0, - p * np.log2(p, where = p > 0), 0)

    return pl2p.sum()


def soft_hard_product_loss(y1, y2):

    hard2 = np.eye(y2.shape[1])[y2.argmax(axis=1),:]
    return 1 - (y1 * hard2).mean(axis=0).sum()

def product_loss(y1, y2):
    
    return 1 - (y1 * y2).mean(axis=0).sum()