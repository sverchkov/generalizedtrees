# Classes for defining constraints used in tests/splits in a decision tree.
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from abc import abstractmethod
from enum import Enum, Flag, auto
from functools import reduce
from typing import Iterable, Protocol, NamedTuple, Any, Union, runtime_checkable
import numpy as np


class Op(Enum):
    EQ = '='
    NEQ = '\u2260'
    LT = '<'
    GT = '>'
    LEQ = '\u2264'
    GEQ = '\u2265'

    def __invert__(self):
        if self is Op.EQ: return Op.NEQ
        if self is Op.NEQ: return Op.EQ
        if self is Op.LT: return Op.GEQ
        if self is Op.GT: return Op.LEQ
        if self is Op.LEQ: return Op.GT
        if self is Op.GEQ: return Op.LT
        raise ValueError(self)

    def test(self, a, b, /):
        if self is Op.EQ: return a == b
        if self is Op.NEQ: return a != b
        if self is Op.LT: return a < b
        if self is Op.GT: return a > b
        if self is Op.LEQ: return a <= b
        if self is Op.GEQ: return a >= b

_orderless = (Op.EQ, Op.NEQ)

_op2bits = {
    Op.EQ: 0b010,
    Op.NEQ: 0b101,
    Op.LT: 0b100,
    Op.GT: 0b001,
    Op.LEQ: 0b110,
    Op.GEQ: 0b011
}

_bits2op = {val: key for key, val in _op2bits.items()}

@runtime_checkable
class Constraint(Protocol):

    @abstractmethod
    def test(self, sample) -> bool:
        raise NotImplementedError()

    def test_matrix(self, data_matrix: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.test, axis=1, arr=data_matrix)

    def __invert__(self) -> 'Constraint':
        return NegatedConstraint(self)


class NegatedConstraint(Constraint):

    def __init__(self, constraint):
        self._constraint = constraint
    
    def test(self, sample):
        return not self._constraint.test(sample)
    
    def test_matrix(self, data_matrix: np.ndarray) -> np.ndarray:
        return ~self._constraint.test(data_matrix)
    
    def __invert__(self):
        return self._constraint
    
    def __str__(self) -> str:
        return f'Not {str(self._constraint)}'


class SimpleConstraint(NamedTuple):
    """
    Single feature axis-alisned constraint.

    Defined in terms of a feature index, operator, and value
    """
    feature: int
    operator: Op
    value: Any

    def test(self, sample):
        return self.operator.test(sample[self.feature], self.value)

    def test_matrix(self, data_matrix: np.ndarray) -> np.ndarray:
        return self.operator.test(data_matrix[:,self.feature], self.value)
    
    def __invert__(self):
        return SimpleConstraint(self.feature, ~self.operator, self.value)
    
    def __str__(self):
        return f'x[{self.feature}] {self.operator.value} {self.value}'
    

# Backwards compatibility aliases
def LEQConstraint(feature_index, value):
    return SimpleConstraint(feature_index, Op.LEQ, value)

def GTConstraint(feature_index, value):
    return SimpleConstraint(feature_index, Op.GT, value)

def EQConstraint(feature_index, value):
    return SimpleConstraint(feature_index, Op.EQ, value)

def NEQConstraint(feature_index, value):
    return SimpleConstraint(feature_index, Op.NEQ, value)


class MofN(Constraint):

    class SearchOperator(Flag):
        INC_M = auto()
        INC_N = auto()
        DEC_M = auto()
        DEC_N = auto()
        INC_NM = INC_N | INC_M

    def __init__(self, m: int, constraints):

        constraints_list = list(constraints)

        # Check for pairs of constraints that cancel each other out
        for i, c1 in enumerate(constraints):
            for c2 in constraints[i+1:]:
                if c1 == ~c2:
                    constraints_list.remove(c1)
                    constraints_list.remove(c2)
                    m -= 1

        # TODO: Check if any of the constraints subsume each other

        self._constraints = tuple(constraints_list)
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

    def test_matrix(self, data_matrix: np.ndarray) -> np.ndarray:

        satisfied = np.zeros(data_matrix.shape[0], dtype=int)

        for c in self._constraints:
            satisfied += c.test_matrix(data_matrix)
        
        return satisfied >= self._m

    def __eq__(self, other):
        return other is self or \
            (isinstance(other, MofN) and
            other.constraints == self.constraints and
            other.m_to_satisfy == self.m_to_satisfy)

    def __bool__(self) -> bool:
        """
        Returns false if the constraint is vacuous
        """
        return self._m > 1
    
    def __str__(self) -> str:
        return f'{self.m_to_satisfy} of {[str(c) for c in self.constraints]}'

    def __repr__(self) -> str:
        return f'MofN({self.m_to_satisfy}, {self.constraints})'

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

            new_m = constraint.m_to_satisfy + (
                1 if operator & MofN.SearchOperator.INC_M else (
                    -1 if operator & MofN.SearchOperator.DEC_M else 0))

            if operator & MofN.SearchOperator.INC_N:
                for atom in constraint_candidates:
                    new_atoms = constraint.constraints + (atom,)
                    yield MofN(new_m, new_atoms)

            elif operator & MofN.SearchOperator.DEC_N:
                atoms = list(constraint.constraints)
                for atom in atoms:
                    yield MofN(new_m, tuple(atoms - {atom}))
            else:
                yield MofN(new_m, constraint.constraints)
        

def vectorize_constraints(constraints: Iterable[SimpleConstraint], dimensions: int):
    upper = np.full(dimensions, np.inf)
    lower = np.full(dimensions, -np.inf)
    upper_eq = np.full(dimensions, True)
    lower_eq = np.full(dimensions, False)

    for constraint in constraints:
        if constraint.operator is Op.GT:
            if lower[constraint.feature] < constraint.value:
                lower[constraint.feature] = constraint.value
        elif constraint.operator is Op.LEQ:
            if upper[constraint.feature] > constraint.value:
                upper[constraint.feature] = constraint.value
        else:
            raise NotImplementedError(f"Cannot vectorize constraint {constraint}")

    return upper, lower, upper_eq, lower_eq


def test_all_x(constraints):
    return lambda x: all([c.test(x) for c in constraints])


def test_all_tuples(constraints):
    return lambda pair: test_all_x(constraints)(pair[0])

def simplify_conjunction(*constraints: SimpleConstraint) -> Union[bool, Iterable[Constraint]]:
    """
    Simplify a conjunction ('and') of constraints into minimal form.

    Currently only supports atomic constraints.
    Reduces a list of constraints to a minimal list of constrains (e.g. if input is x_1 < 2 and
    x_1 < 3, the output will only by x_1 < 2)
    """

    # TODO: Need to think about how to do this.

    raise NotImplementedError("Not implemented yet.")

def simplify_disjunction(*constraints):
    """
    Simplify a disjunction ('or') of constraints into minimal form.

    Currently only supports atomic constraints.
    Reduces a list of constraints to a minimal list of constrains (e.g. if input is x_1 < 2 and
    x_1 < 3, the output will only by x_1 < 3)
    """
    raise NotImplementedError("Not implemented yet.")


def test(constraints: Iterable[Constraint], data_matrix):

    n, _ = data_matrix.shape

    return reduce(np.multiply, map(lambda c: c.test_matrix(data_matrix), constraints), np.ones(n, dtype=np.bool))
