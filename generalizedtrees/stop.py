# Implementations of stopping criteria
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov


from abc import abstractmethod
from typing import Iterable, Protocol
from generalizedtrees.tree import Tree


# Stopping criteria learner components:

## Interface definitions

class LocalStopLC(Protocol):

    @abstractmethod
    def check(self, node) -> bool:
        raise NotImplementedError


class GlobalStopLC(Protocol):

    @abstractmethod
    def check(self, tree: Tree) -> bool:
        raise NotImplementedError


## Composite criteria

class LocalStopConjunctionLC(LocalStopLC):

    subcriteria: Iterable[LocalStopLC] = []

    def __init__(self, *subcriteria):
        self.subcriteria = subcriteria

    def check(self, node) -> bool:

        for criterion in self.criteria:
            if not criterion.check(node): return False
        
        return True


class LocalStopDisjunctionLC(LocalStopLC):

    subcriteria: Iterable[LocalStopLC] = []

    def __init__(self, *subcriteria):
        self.subcriteria = subcriteria

    def check(self, node) -> bool:

        for criterion in self.criteria:
            if criterion.check(node): return True
        
        return False


class GlobalStopConjunctionLC(GlobalStopLC):

    subcriteria: Iterable[GlobalStopLC] = []

    def __init__(self, *subcriteria):
        self.subcriteria = subcriteria

    def check(self, tree: Tree) -> bool:

        for criterion in self.criteria:
            if not criterion.check(tree): return False
        
        return True


class GlobalStopDisjunctionLC(GlobalStopLC):

    subcriteria: Iterable[GlobalStopLC] = []

    def __init__(self, *subcriteria):
        self.subcriteria = subcriteria

    def check(self, tree: Tree) -> bool:

        for criterion in self.criteria:
            if criterion.check(tree): return True
        
        return False


## General stopping criteria

class NeverStopLC(LocalStopLC, GlobalStopLC):
    def check(*args, **kwargs) -> bool:
        return False


## Local stopping criteria

class LocalStopDepthLC(LocalStopLC):

    depth: int

    def __init__(self, max_depth):
        self.depth = max_depth
    
    def check(self, node):
        return node.depth >= self.depth

# Todo: saturation test


## Global stopping criteria

class GlobalStopTreeSizeLC(GlobalStopLC):

    size: int

    def __init__(self, max_size):
        self.size = max_size
    
    def check(self, tree: Tree):
        return len(tree) >= self.size
