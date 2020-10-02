# Base class hierarchy for tree models
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

from functools import total_ordering
from abc import ABC, abstractmethod
from generalizedtrees.tree import tree_to_str, Tree
from generalizedtrees.queues import CanPushPop
from typing import Protocol, Optional, Tuple, Any, Iterator
import numpy as np
import pandas as pd


class AbstractTreeBuilder(Protocol):
    """
    Abstract class for tree builders.

    Mostly used for type annotations.
    """

    @abstractmethod
    def build_tree(self) -> Tree:
        raise NotImplementedError

    def prune_tree(self, tree) -> Tree: # Note: not abstract; no-op is a default pruning strategy
        return tree


class SplitTest(Protocol):
    """
    Base abstract class for splits.

    A split is defined as a test with integer outcomes and a set of constraints each of which
    corresponds to an outcome of the test.
    """

    @abstractmethod
    def pick_branches(self, data_frame):
        raise NotImplementedError


    @property
    @abstractmethod
    def constraints(self):
        raise NotImplementedError


class GreedyTreeBuilder(AbstractTreeBuilder):
    """
    Greedy tree building strategy
    """

    def build_tree(self):

        root = self.create_root()
        tree = Tree([root])

        queue = self.new_queue()
        queue.push((root, 'root'))

        while queue and not self.global_stop(tree):

            node, ptr = queue.pop()

            if not self.local_stop(tree.node(ptr)):

                node.split = self.construct_split(node)

                if node.split is not None:
                    for child in self.generate_children(node):
                        child_ptr = tree.add_node(child, parent_key=ptr)
                        queue.push((child, child_ptr))
        
        return tree
    
    @abstractmethod
    def create_root(self) -> Any:
        pass

    @abstractmethod
    def generate_children(self, parent) -> Iterator:
        pass

    @abstractmethod
    def new_queue(self) -> CanPushPop:
        pass

    @abstractmethod
    def construct_split(self, node) -> SplitTest:
        pass

    @abstractmethod
    def global_stop(self, tree: Tree) -> bool:
        pass

    @abstractmethod
    def local_stop(self, node) -> bool:
        pass


## Decorator for making a class orderable
def order_by(*attrs):
    """
    Make a class orderable by a set of its attributes.

    Class decorator.

    Parameters are the names of attributes in order of priority.
    This defines all comparison operators (and __eq__) based on the values of the attributes.
    (Actually we only define __lt__ and __eq__ and pass to functools.total_ordering to define the rest.)
    """

    def decorate(cls):

        def eq(self, other):
            return all(getattr(self, a) == getattr(other, a) for a in attrs)
        
        def lt(self, other):

            for a in attrs:
                ours = getattr(self, a)
                theirs = getattr(other, a)
                if ours < theirs: return True
                elif theirs > ours: return False
            
            return False
        
        setattr(cls, '__lt__', lt)
        setattr(cls, '__eq__', eq)

        return total_ordering(cls)
    
    return decorate
