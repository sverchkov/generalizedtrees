# Packaged queue and stack classes
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from typing import Generic, Protocol, TypeVar
from abc import abstractmethod
from heapq import heappush, heappop
from collections import deque

T = TypeVar('T')

class CanPushPop(Protocol, Generic[T]):

    @abstractmethod
    def push(self, item: T) -> None:
        raise NotImplementedError

    @abstractmethod
    def pop(self) -> T:
        raise NotImplementedError

class Heap(list, CanPushPop):

    def push(self, item):
        heappush(self, item)
    
    def pop(self):
        return heappop(self)

class Stack(deque, CanPushPop):

    def push(self, item):
        self.append(item)

class Queue(Stack):

    def pop(self):
        return super().popleft()
    
    def popright(self):
        return super().pop()
