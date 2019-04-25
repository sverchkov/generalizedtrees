# Trepan-like mimic tree learner
#
# Based on Trepan (Craven and Shavlik 1996)
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

from abc import abstractmethod
from generalizedtrees.core import Constraint
import numpy as np


class SingleFeatureConstraint(Constraint):

    def __init__(self, feature_index):
        self._feature = feature_index

    @property
    def feature(self):
        return self._feature

    def test(self, sample):
        return self.test_value(sample[self._feature])

    @abstractmethod
    def test_value(self, value):
        pass


class LEQConstraint(SingleFeatureConstraint):

    def __init__(self, feature_index, value):
        self._value = value
        super().__init__(feature_index)

    @property
    def value(self):
        return self._value

    def test_value(self, value):
        return value <= self._value

    def __invert__(self):
        return GTConstraint(self.feature, self._value)

    def __repr__(self):
        return f"[{self.feature}]<={self._value}"


class GTConstraint(SingleFeatureConstraint):

    def __init__(self, feature_index, value):
        self._value = value
        super().__init__(feature_index)

    @property
    def value(self):
        return self._value

    def test_value(self, value):
        return value > self._value

    def __invert__(self):
        return LEQConstraint(self.feature, self._value)

    def __repr__(self):
        return f"[{self.feature}]>{self._value}"


def vectorize_constraints(constraints, dimensions):
    upper = np.full(dimensions, np.inf)
    lower = np.full(dimensions, -np.inf)
    upper_eq = np.full(dimensions, True)
    lower_eq = np.full(dimensions, False)

    for constraint in constraints:
        if isinstance(constraint, GTConstraint):
            if lower[constraint.feature] < constraint.value:
                lower[constraint.feature] = constraint.value
        elif isinstance(constraint, LEQConstraint):
            if upper[constraint.feature] > constraint.value:
                upper[constraint.feature] = constraint.value
        else:
            raise NotImplementedError(f"Cannot vectorize constraint {constraint}")

    return upper, lower, upper_eq, lower_eq
