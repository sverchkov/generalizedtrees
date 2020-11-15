# Scoring functions
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

    return 1 - np.sum(p*p)


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