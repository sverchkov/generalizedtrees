# Tests for our tree data structure
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

from collections import deque
from typing import Iterable, Any, List, NamedTuple, Union

class TreeNode:
    """
    An interface for manipulating a tree via its nodes
    """

    def __init__(self, tree: 'Tree', index: int, parent: int, depth: int, content: Any):

        # Private members
        self._tree: Tree = tree
        self._index: int = index
        self._parent: int = parent
        self._children: Iterable[int] = []
        self._depth: int = depth

        # Public members
        self.content = content

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def tree(self) -> 'Tree':
        return self._tree

    @property
    def is_leaf(self) -> int:
        return not self._children

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, key) -> Union['TreeNode', Iterable['TreeNode']]:
        """
        We use slices to access child nodes
        """
        return self._tree[self._children[key]]

    def add_children(self, children_contents):
        n = len(self._tree)
        k = len(children_contents)
        idxs = range(n, n+k)
        new_depth = self._depth + 1
        self._tree._nodes.extend([
            TreeNode(
                tree = self._tree,
                index = i,
                parent = self._index,
                depth = new_depth,
                content = c)
            for i, c in zip(idxs, children_contents)
        ])
        self._children.extend(idxs)
        if new_depth > self._tree._tree_depth:
            self._tree._tree_depth = new_depth
    
    def parent(self):
        return self._tree[self._parent]


class Tree:
    """
    A container-like tree data structure
    """

    def __init__(self, content):
        self._nodes = [TreeNode(self, 0, -1, 0, content)]
        self._tree_depth = 0

    @property
    def depth(self) -> int:
        return self._tree_depth

    @property
    def size(self) -> int:
        return len(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __getitem__(self, key) -> Union[TreeNode, Iterable[TreeNode]]:

        if key == 'root':
            return self._nodes[0]
        
        if isinstance(key, int) or isinstance(key, slice):
            return self._nodes[key]

        return [self._nodes[i] for i in key]

    @property
    def root(self) -> TreeNode:
        return self._nodes[0]


def tree_to_str(tree: Tree) -> str:

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
        
        result += str(node.content)
        if stack:
            result += endline
    
    return result

