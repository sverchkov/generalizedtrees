# Sampling utilities
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

from generalizedtrees.constraints import vectorize_constraints


def univariate_gaussian_rejection_sample(
        mu: float = 0,
        sigma: float = 1,
        n: int = 1,
        upper: float = np.inf,
        lower: float = -np.inf,
        upper_eq: bool = False,
        lower_eq: bool = False) -> np.ndarray:

    sampled: int = 0
    x: np.ndarray = np.zeros(n)

    while sampled < n:

        # Sampling
        new_x: np.ndarray = np.random.normal(mu, sigma, n - sampled)

        # Rejection
        keep: np.ndarray = np.logical_and(
            (np.less_equal if upper_eq else np.less)(new_x, upper),
            (np.greater_equal if lower_eq else np.greater)(new_x, lower))

        # Update
        sampled_new: int = sampled + sum(keep)
        x[sampled:sampled_new] = new_x[keep]
        sampled = sampled_new

    return x


_vectorized_gaussian_rejection_sample = np.vectorize(
    univariate_gaussian_rejection_sample,
    otypes=[np.ndarray],
    excluded='n')


def gaussian_rejection_sample(mu, sigma, n, constraints):

    upper, lower, upper_eq, lower_eq = vectorize_constraints(constraints)

    return np.column_stack(_vectorized_gaussian_rejection_sample(mu, sigma, n, upper, lower, upper_eq, lower_eq))
