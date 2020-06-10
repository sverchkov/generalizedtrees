# Classes for defining constraints used in tests/splits in a decision tree.
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
from enum import Flag, auto
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

    def __eq__(self, other):
        return other is self or \
            (isinstance(other, LEQConstraint) and
            other.feature == self.feature and
            other.value == self.value)


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

    def __eq__(self, other):
        return other is self or \
            (isinstance(other, GTConstraint) and
            other.feature == self.feature and
            other.value == self.value)


class EQConstraint(SingleFeatureConstraint):

    def __init__(self, feature_index, value):
        self._value = value
        super().__init__(feature_index)

    @property
    def value(self):
        return self._value

    def test_value(self, value):
        return value == self.value

    def __invert__(self):
        return NEQConstraint(self.feature, self.value)

    def __repr__(self):
        return f"[{self.feature}]=={self.value}"

    def __eq__(self, other):
        return other is self or \
            (isinstance(other, EQConstraint) and
            other.feature == self.feature and
            other.value == self.value)


class NEQConstraint(SingleFeatureConstraint):

    def __init__(self, feature_index, value):
        self._value = value
        super().__init__(feature_index)

    @property
    def value(self):
        return self._value

    def test_value(self, value):
        return value != self.value

    def __invert__(self):
        return EQConstraint(self.feature, self.value)

    def __repr__(self):
        return f"[{self.feature}]!={self.value}"

    def __eq__(self, other):
        return other is self or \
            (isinstance(other, NEQConstraint) and
            other.feature == self.feature and
            other.value == self.value)


class MofN(Constraint):

    class SearchOperator(Flag):
        INC_M = auto()
        INC_N = auto()
        INC_NM = INC_N | INC_M

    def __init__(self, m, constraints):

        # TODO: Check if any of the constraints cancel each other out

        self._constraints = constraints
        self._m = m

    @property
    def constraints(self):
        return self._constraints

    @property
    def m_to_satisfy(self):
        return self._m

    @property
    def number_of_constraints(self):
        return len(self._constraints)

    def test(self, sample):

        satisfied: int = 0

        for c in self._constraints:
            satisfied += c.test(sample)  # Implicitly converting bool to 0/1
            if satisfied >= self._m:
                return True

        return False

    def __eq__(self, other):
        return other is self or \
            (isinstance(other, MofN) and
            other.constraints == self.constraints and
            other.m_to_satisfy == self.m_to_satisfy)


    @staticmethod
    def neighboring_tests(
        constraint,
        constraint_candidates,
        search_operators=[SearchOperator.INC_N, SearchOperator.INC_NM]):
        """
        Expand an atomic or an m-of-n constraint by one step using the search operators specified
        and the set of atomic constraints specified.
        """

        # Wrap input constraint as 1-of-1 if it's not an m-of-n constraint:
        if not isinstance(constraint, MofN):
            constraint = MofN(1, (constraint,))
        
        for operator in search_operators:

            new_m = constraint.m_to_satisfy + \
                (1 if operator & MofN.SearchOperator.INC_M else 0)

            if operator & MofN.SearchOperator.INC_N:
                for atom in constraint_candidates:
                    new_atoms = constraint.constraints + (atom,)

                    yield MofN(new_m, new_atoms)
            
            else:
                yield MofN(new_m, constraint.constraints)
        

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
