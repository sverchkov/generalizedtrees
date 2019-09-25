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
from generalizedtrees.core import test_all_x


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

    upper, lower, upper_eq, lower_eq = vectorize_constraints(constraints, len(mu))

    return np.column_stack(_vectorized_gaussian_rejection_sample(mu, sigma, n, upper, lower, upper_eq, lower_eq))


def rejection_sample_generator(generator):

    def constrained_generator(n, constraints):

        tester = test_all_x(constraints)
        sampled = []

        while len(sampled) < n:
            sampled.extend([x for x in generator(n-len(sampled)) if tester(x)])

        return sampled

    return constrained_generator


def agnostic_univariate_sample(n: int = 1, lower: float = -np.inf, upper: float = np.inf, scale: float = 1):
    """
    Sample a single feature agnostically:
    - from standard Cauchy (0 mode) if unconstrained
    - from abs(Cauchy) if bounded on one side
    - from uniform if bounded on both sides
    :param n: Number of samples
    :param lower: Lowerbound (-inf for unbounded)
    :param upper: Upperbound (inf for unbounded)
    :param scale: Scale parameter for unbounded distributions
    :return:
    """

    unbounded_above = np.isinf(upper)
    unbounded_below = np.isinf(lower)

    if unbounded_above or unbounded_below:
        r = np.random.standard_cauchy(n)*scale  # Unbounded means we'll use the Cauchy somehow
        if not unbounded_below:
            return lower + abs(r)
        elif not unbounded_above:
            return upper - abs(r)
        else:
            return r
    else:
        return np.random.uniform(lower, upper, n)


_vectorized_agnostic_sample = np.vectorize(
    agnostic_univariate_sample,
    otypes=[np.ndarray],
    excluded='n')


def agnostic_numeric_sample(n, dim, constraints):
    """
    Samples a fully numeric feature matrix that satisfies the constraints.
    Sampling strategy is described in :func:`agnostic_univariate_sample`
    :param n: Number of samples
    :param dim: Number of features
    :param constraints: Iterable over Constraints
    :return: n x dim feature matrix that satisfies the constraints
    """

    upper, lower, _, _ = vectorize_constraints(constraints, dim)

    return np.column_stack(_vectorized_agnostic_sample(n, lower, upper))
