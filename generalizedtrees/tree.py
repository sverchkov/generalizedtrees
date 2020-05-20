from collections import deque
from typing import Iterable, Any, List, NamedTuple

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
    def tree(self) -> Tree:
        return self._tree

    @property
    def is_leaf(self) -> int:
        return not self._children

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, key) -> Iterable['TreeNode']:
        """
        We use slices to access child nodes
        """
        return self._tree[self._children[key]]

    def set_children(self, children_contents):
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
            for c, i in zip(idxs, children_contents)
        ])
        self._children = idxs
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

    def __getitem__(self, key) -> TreeNode:
        if key == 'root':
            return self._nodes[0]
        
        return self._nodes[key]

    @property
    def root(self) -> TreeNode:
        return self._nodes[0]


def tree_to_str(tree: Tree) -> str:

    # Constants for tree drawing. Defining them here in case we want to customize later.
    space = ' '
    trunk = '|'
    mid_branch = '+'
    last_branch = '+'
    endline = '\n'

    result:str = ''
    stack = deque(tree.root)
    continuing = [0 for _ in range(tree.depth)]

    while stack:

        node = stack.pop()
        stack.extend(node[:])
        continuing[node.depth] += len(node)

        if node.depth > 0:
            for d in range(node.depth-1):
                result += trunk if continuing[d] else space
            continuing[node.depth-1] -= 1
            result += mid_branch if continuing[node.depth-1] else last_branch
        
        result += str(node.content) + endline

