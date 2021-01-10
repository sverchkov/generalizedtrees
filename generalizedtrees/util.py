# Utility functions
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from dataclasses import dataclass
from functools import total_ordering
from typing import Any


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


## Dataclass for scored items
@order_by('score')
@dataclass
class ScoredItem:
    score: float
    item: Any