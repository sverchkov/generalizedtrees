# Tree data structure
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from collections.abc import Collection
from collections import deque
from typing import Iterable, Any
from logging import getLogger

logger = getLogger()


class Tree(Collection):
    """
    A container-like tree data structure.
    """

    class Node:

        def __init__(self, tree: 'Tree', index: int, item: Any, depth: int, parent: int):
            self.tree = tree
            self.item = item
            self._depth = depth
            self._parent = parent
            self._index = index
            self._children = []
        
        def parent(self):
            return self.tree.node(self._parent)

        @property
        def depth(self):
            return self._depth
        
        @property
        def is_root(self) -> bool:
            return self._depth == 0

        @property
        def is_leaf(self) -> bool:
            return not self._children

        def __len__(self) -> int:
            return len(self._children)

        def __getitem__(self, key):
            """
            We use slices to access child nodes
            """
            return self.tree.node(self._children[key])

    def __init__(self, contents = []):
        """
        Create a tree

        contents: heterogeneous iterable specifying the tree. The list is interpreted as
        follows:
            - The first item is the item at the root
            - Subsequent items are subtrees following the same convention.
            i.e. ['A', ['B', ['D']], ['C']] would initialize a tree with root
            'A', children 'B' and 'C', and 'D' as a child of 'B'.
            - Enclosing list may be omitted for leaves.
        """
        self._nodes = []
        self._tree_depth = -1

        # Populate tree
        if not isinstance(contents, Iterable) or contents:
            stack = deque([(-1, contents)])
            while stack:
                parent, subtree = stack.pop()
                if not isinstance(subtree, Iterable):
                    self.add_node(subtree, parent)
                else:
                    it = iter(subtree)
                    index = self.add_node(next(it), parent)
                    stack.extend((index, subtr) for subtr in it)

    @property
    def depth(self) -> int:
        return self._tree_depth

    def __len__(self) -> int:
        return len(self._nodes)

    def _single_index(self, key):
        return isinstance(key, int) or key == 'root' 

    def _1index(self, key):
        k = 0 if key == 'root' else key
        if isinstance(k, int) and k >=0 and k < len(self._nodes):
            return k
        else:
            raise IndexError(
                f'Key {key} is out of bounds for tree of '
                f'size {len(self._nodes)} or is not a single value.')

    def __getitem__(self, key):
        if self._single_index(key):
            return self.node(key).item
        else:
            return (n.item for n in self.node(key))

    def node(self, key):
        if self._single_index(key):
            return self._nodes[self._1index(key)]
        elif isinstance(key, slice):
            return self._nodes[key]
        elif isinstance(key, Iterable):
            return (self._nodes[self._1index(k)] for k in key)
        raise TypeError(f'Tree indices must be strings, integers, slices, or iterable')

    @property
    def root(self):
        return self.node(0)
    
    def add_node(self, item, parent_key=-1) -> int:
        if parent_key == -1:
            if self._nodes:
                raise ValueError('Attempted to replace existing root node.')
            else:
                self._nodes.append(Tree.Node(self, 0, item, 0, -1))
                self._tree_depth = 0
        else:
            self.add_children([item], parent_key)
        
        # Assuming that node was ended to the end
        return len(self._nodes)-1

    def add_children(self, items, parent_key):

        # Check parent index
        parent_key = self._1index(parent_key)

        # Get parent node object
        parent: Tree.Node = self._nodes[parent_key]

        # Determine depth of child nodes
        depth = parent.depth + 1

        # Determine child indeces in nodes array
        indeces = range(len(self._nodes), len(self._nodes)+len(items))

        # Create child nodes and add to nodes array
        self._nodes.extend(
            Tree.Node(self, index, item, depth, parent_key)
            for index, item in zip(indeces, items))
        
        # Register children with parent
        parent._children.extend(indeces)

        # Update tree depth
        if depth > self._tree_depth:
            self._tree_depth = depth
    
    def __iter__(self):
        """
        Depth-first iteration
        """
        if (self):
            stack = deque([0])

            while stack:
                n = stack.pop()
                stack.extend(self._nodes[n]._children)
                yield self._nodes[n].item

    def __contains__(self, item):
        return any(item is node.item or item == node.item for node in self._nodes)


def tree_to_str(tree: Tree, content_str = lambda x: str(x)) -> str:

    # Constants for tree drawing. Defining them here in case we want to customize later.
    space = '   '
    trunk = '|  '
    mid_branch = '+--'
    last_branch = '+--'
    endline = '\n'

    result:str = ''
    stack = deque([tree.root])
    continuing = [0 for _ in range(tree.depth+1)]

    while stack:

        node = stack.pop()
        stack.extend(node[:])
        continuing[node.depth] += len(node)

        if node.depth > 0:
            for d in range(node.depth-1):
                result += trunk if continuing[d] else space
            continuing[node.depth-1] -= 1
            result += mid_branch if continuing[node.depth-1] else last_branch
        
        result += content_str(node.item)
        if stack:
            result += endline
    
    return result

