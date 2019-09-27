# Generalized Tree Models
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


from typing import List, Tuple
from abc import ABC, abstractmethod
from collections import deque
from numpy import ndarray, empty, apply_along_axis
from sklearn.exceptions import NotFittedError
import logging

logger = logging.getLogger(__name__)


class Constraint(ABC):

    @abstractmethod
    def test(self, sample):
        pass


class Node:

    def __init__(self, constraint: Constraint = None, parent: "Node" = None):
        self.parent = parent
        self.constraint = constraint
        self.model = None

    @property
    def all_constraints(self) -> () or Tuple[Constraint]:
        if self.parent:
            if self.constraint is None:
                logger.error(f'Node {self} with parent {self.parent}. This is a bad constraint configuration.')
            return self.parent.all_constraints + (self.constraint,)
        elif self.constraint is not None:
            return self.constraint,  # Trailing comma because this is a tuple
        else:
            return ()

    def __repr__(self):
        return f'({self.constraint}: {self.model})'


class ChildSelector:

    def __init__(self, children: List[Node]):
        self.children = children

    def predict(self, data: ndarray) -> ndarray:
        assert isinstance(data, ndarray), "Data must be a numpy array"
        assert data.ndim == 2, "Data matrix must be 2D"
        n = data.shape[0]

        if n <= 0:
            return empty(0, dtype=object)

        else:

            prediction = None

            for c in self.children:

                indexes = apply_along_axis(c.constraint.test, 1, data)  # Find instances selected by child

                if indexes.any():

                    pred = c.model.predict(data[indexes])  # Get child predictions

                    if prediction is None:  # Make sure we have the correct result array type
                        prediction = empty(n, dtype=pred.dtype)

                    prediction[indexes] = pred  # Insert result at the correct indexes

            return prediction

    def __repr__(self):
        return self.children.__repr__()


class NodeQueue:

    def __init__(self):
        self.q = deque()

    def append(self, x):
        self.q.append(x)

    def extend(self, xs):
        self.q.extend(xs)

    def pop(self):
        return self.q.popleft()

    def __len__(self):
        return len(self.q)


class AbstractTreeEstimator(ABC):

    def __init__(
            self,
            sequential_access_data_structure_factory=NodeQueue):
        """
        The abstract tree classifier is defined by
        :param sequential_access_data_structure_factory: A  factory that produces a data structure such as a stack,
        queue, or priority queue that determines the order in which the tree is built. For a priority queue the rule for
        comparing nodes is also communicated through this function. The data structure must implement append (for one
        element), extend (for a list of elements) and pop.
        """
        assert callable(sequential_access_data_structure_factory),\
            "Sequential access data structure factory must be callable"
        self.root_: Node
        self.sequential_access_data_structure_factory = sequential_access_data_structure_factory

    @abstractmethod
    def best_split(self, constraints):
        pass

    @abstractmethod
    def leaf_predictor(self, constraints):
        pass

    @abstractmethod
    def check_data_for_predict(self, data):
        """
        :param data: Data to check
        Should raise errors if the data passed to predict is bad
        :return: Checked data
        """
        pass

    def best_split_of_node(self, node: Node):
        return self.best_split(node.all_constraints)

    def leaf_predictor_of_node(self, node: Node):
        return self.leaf_predictor(node.all_constraints)

    def build(self):
        self.root_ = Node()
        nodes = self.sequential_access_data_structure_factory()
        nodes.append(self.root_)

        while nodes:
            parent = nodes.pop()
            split = self.best_split_of_node(parent)

            logger.log(5, f'building: {self}')
            logger.log(5, f'best split: {split}')

            if len(split) > 1:  # Splitting
                children = [Node(s, parent) for s in split]
                logger.log(5, f'setting children: {children}')
                parent.model = ChildSelector(children)
                nodes.extend(children)

            else:  # It's a leaf
                parent.model = self.leaf_predictor_of_node(parent)

    def predict(self, data: ndarray, enforce_finite=True):
        if not hasattr(self, 'root_'):
            raise NotFittedError("Tried to predict before building tree (usually through fit)")
        result = self.root_.model.predict(self.check_data_for_predict(data))
        logger.log(5, f"Predicting {result} ({type(result)}; {result.dtype})")
        return result

    def predict_instance(self, sample):  # May be deprecated later on
        return self.root_.model.predict(sample.reshape((1, -1)))[0]

    def __repr__(self):
        return f"Tree: {self.root_}"


def test_all_x(constraints):
    return lambda x: all([c.test(x) for c in constraints])


def test_all_tuples(constraints):
    return lambda pair: test_all_x(constraints)(pair[0])
