# Tree growing algorithms
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from typing import Callable

from generalizedtrees.givens import GivensLC
from generalizedtrees.node import NodeBuilderLC
from generalizedtrees.queues import CanPushPop
from generalizedtrees.split import SplitConstructorLC, DefaultSplitConstructorLC
from generalizedtrees.stop import GlobalStopLC, LocalStopLC, NeverStopLC
from generalizedtrees.tree import Tree


###########################
# Tree builder components #
###########################

# Future: define interface protocol if we will ever have other builder strategies

class GreedyBuilderLC:
    """
    Greedy tree building strategy
    """

    new_queue: Callable[..., CanPushPop]
    node_builder: NodeBuilderLC
    splitter: SplitConstructorLC
    global_stop: GlobalStopLC = NeverStopLC()
    local_stop: LocalStopLC = NeverStopLC()

    def __init__(self):
        self.splitter = DefaultSplitConstructorLC()
    
    def initialize(self, givens: GivensLC) -> None:
        self.node_builder.initialize(givens)
        self.splitter.initialize(givens)

    def build_tree(self) -> Tree:

        # Init root
        root, data, y = self.node_builder.create_root()
        tree = Tree([root])

        # Init queue
        queue = self.new_queue()
        queue.push((root, 'root', data, y))

        # Init node expansion order tracking
        node_number: int = 0

        # Queue-order expansion:
        while queue and not self.global_stop.check(tree):

            node, ptr, data, y = queue.pop()
            node.node_number = node_number
            node_number += 1

            if not self.local_stop.check(tree.node(ptr), data, y):

                node.split = self.splitter.construct_split(node, data, y)

                if node.split:
                    for child, c_data, c_y in self.node_builder.generate_children(node, data, y):
                        child_ptr = tree.add_node(child, parent_key=ptr)
                        queue.push((child, child_ptr, c_data, c_y))
        
        return tree
    
    def prune_tree(self, tree: Tree):
        return tree
