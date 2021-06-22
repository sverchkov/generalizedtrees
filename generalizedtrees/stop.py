# Implementations of stopping criteria
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov


from abc import abstractmethod, ABC
from logging import getLogger, Logger
from typing import Iterable, Protocol

from generalizedtrees.tree import Tree

logger: Logger = getLogger()

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

        for criterion in self.subcriteria:
            if not criterion.check(node): return False
        
        return True


class LocalStopDisjunctionLC(LocalStopLC):

    subcriteria: Iterable[LocalStopLC] = []

    def __init__(self, *subcriteria):
        self.subcriteria = subcriteria

    def check(self, node) -> bool:

        for criterion in self.subcriteria:
            if criterion.check(node): return True
        
        return False


class GlobalStopConjunctionLC(GlobalStopLC):

    subcriteria: Iterable[GlobalStopLC] = []

    def __init__(self, *subcriteria):
        self.subcriteria = subcriteria

    def check(self, tree: Tree) -> bool:

        for criterion in self.subcriteria:
            if not criterion.check(tree): return False
        
        return True


class GlobalStopDisjunctionLC(GlobalStopLC):

    subcriteria: Iterable[GlobalStopLC] = []

    def __init__(self, *subcriteria):
        self.subcriteria = subcriteria

    def check(self, tree: Tree) -> bool:

        for criterion in self.subcriteria:
            if criterion.check(tree): return True
        
        return False


## General stopping criteria

class NeverStopLC(LocalStopLC, GlobalStopLC):
    def check(self, *args, **kwargs) -> bool:
        return False


## Local stopping criteria

class LocalStopDepthLC(LocalStopLC):

    def __init__(self, max_depth: int):
        self.depth: int = max_depth
    
    def check(self, node):
        if node.depth >= self.depth:
            logger.debug(f'Local stopping criterion met: node depth {node.depth} meets depth limit {self.depth}')
            return True
        return False

class LocalStopSaturation(LocalStopLC):

    def __init__(self, saturation: float = 1, training_only: bool = False):
        self.saturation: float = saturation
        self.training_only: bool = training_only

    def check(self, node) -> bool:
        item = node.item
        y = item.y
        if self.training_only:
            if hasattr(item, 'n_training'):
                if item.n_training <= 1:
                    logger.debug(f'Local stopping criterion met: {item.n_training} samples for saturation test.')
                    return True
                else:
                    y = item.y[:item.n_training, :]
            else:
                logger.warn(
                    'Node saturation stopping criterion "training_only" flag is mean to be '
                    'used with model translation, but is being used with node objects that '
                    'do not support it.')

        saturation = y.mean(axis=0).max()

        if saturation >= self.saturation:
            logger.debug(f'Local stopping criterion met: node saturation={saturation} meets threshold={self.saturation}.')
            return True

        return False



## Global stopping criteria

class GlobalStopTreeSizeLC(GlobalStopLC):

    def __init__(self, max_size):
        self.size: int = max_size
    
    def check(self, tree: Tree):
        size = len(tree)
        if size >= self.size:
            logger.debug(f'Global stopping criterion met: tree size {size} meets size limit {self.size}.')
            return True
        return False
