# Tree data structure
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

from abc.collections import Collection, Iterator
from collections import deque
from typing import Iterable, Any, List, NamedTuple, Union
from logging import getLogger

logger = getLogger()

class TreeNode:
    """
    A base class for the nodes of a tree.
    This class defines all the bookkeeping variables needed to maintain a tree.
    Meant to be extended to classes that hold actual content at nodes.
    """

    def __init__(self, tree: 'Tree' = None, index: int = 0, parent: int = -1, depth: int = 0):

        # Private members
        self._tree: Tree = tree
        self._index: int = index
        self._child_index: int = -1
        self._parent: int = parent
        self._children: Iterable[int] = []
        self._depth: int = depth

    def plant_tree(self):
        """
        Make a tree rooted at this node.
        This node becomes attached to the new tree.
        """
        #if self._tree is not None: raise some exception
        self._tree = Tree(self)
        return self._tree

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def tree(self) -> 'Tree':
        # if self._tree is None: raise some exception
        return self._tree

    @property
    def is_root(self) -> bool:
        return self.depth == 0

    @property
    def is_leaf(self) -> bool:
        return not self._children

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, key) -> Union['TreeNode', Iterable['TreeNode']]:
        """
        We use slices to access child nodes
        """
        try:
            return self._tree[self._children[key]]
        except:
            logger.fatal(
                'Something went wrong in getting a child node:\n'
                f'key: {key}\n'
                f'index of calling node: {self._index}\n'
                f'indices of children: {self._children}\n'
                f'tree size: {len(self._tree._nodes)}\n')
            raise

    def add_child(self, child: 'TreeNode'):
        # TODO: Add check that child isn't already in some tree
        # TODO: Add check that child doesn't have children set
        child._tree = self._tree
        child._children = []
        child._parent = self._index
        n = len(self._tree)
        child._index = n
        child._child_index = len(self._children)
        self._children.append(n)
        self._tree._nodes.append(child)
        d = self._depth + 1
        child._depth = d
        if self._tree.depth < d:
            self._tree._tree_depth = d
    
    @property
    def parent(self):
        return self._tree[self._parent]


class Tree(Collection):
    """
    A container-like tree data structure.
    """

    class Node:

        def __init__(self, tree: Tree, index: int, item: Any, depth: int, parent: int):
            self.tree = tree
            self.item = item
            self._depth = depth,
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
            try:
                return self.tree.node(self._children[key])
            except:
                logger.fatal(
                    'Something went wrong in getting a child node:\n'
                    f'key: {key}\n'
                    f'index of calling node: {self._index}\n'
                    f'indices of children: {self._children}\n'
                    f'tree size: {len(self.tree._nodes)}\n')
                raise

    def __init__(self):
        self._nodes = []
        self._tree_depth = -1

    @property
    def depth(self) -> int:
        return self._tree_depth

    def __len__(self) -> int:
        return len(self._nodes)

    def _index(self, key):
        k = 0 if key == 'root' else key
        if isinstance(k, int) and k >=0 and k < len(self._nodes):
            return k
        else:
            raise IndexError(f'Key {key} is out of bounds for tree of size {len(self._nodes)}')

    def __getitem__(self, key):
        return self._nodes[self._index(key)].item

    def node(self, key):
        return self._nodes[self._index(key)]

    @property
    def root(self):
        return self.node(0)
    
    def add_node(self, item, parent_key=-1):
        if parent_key < 0:
            if self:
                raise ValueError('Attempted to replace existing root node.')
            else:
                self._nodes.append(Tree.Node(self, 0, item, 0, -1))
                self._tree_depth = 0
        else:
            self.add_children([item], parent_key)

    def add_children(self, items, parent_key):

        # Check parent index
        parent_key = self._index(parent_key)

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
        stack.extend(reversed(node[:]))
        continuing[node.depth] += len(node)

        if node.depth > 0:
            for d in range(node.depth-1):
                result += trunk if continuing[d] else space
            continuing[node.depth-1] -= 1
            result += mid_branch if continuing[node.depth-1] else last_branch
        
        result += content_str(node)
        if stack:
            result += endline
    
    return result

