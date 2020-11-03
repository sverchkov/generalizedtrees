# Tree growing algorithms
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

from generalizedtrees.stop import GlobalStopLC, LocalStopLC
from typing import Callable, Iterable

from generalizedtrees.base import SplitTest
from generalizedtrees.queues import CanPushPop
from generalizedtrees.tree import Tree

# Builder components

# Future: define interface protocol if we will ever have other builder strategies

class GreedyBuilderLC:
    """
    Greedy tree building strategy
    """

    new_queue: Callable[..., CanPushPop]
    create_root: Callable
    generate_children: Callable[..., Iterable]
    construct_split: Callable[..., SplitTest]
    global_stop: GlobalStopLC
    local_stop: LocalStopLC

    def build_tree(self):

        root = self.create_root()
        tree = Tree([root])

        queue = self.new_queue()
        queue.push((root, 'root'))

        while queue and not self.global_stop.check(tree):

            node, ptr = queue.pop()

            if not self.local_stop.check(tree.node(ptr)):

                node.split = self.construct_split(node)

                if node.split is not None:
                    for child in self.generate_children(node):
                        child_ptr = tree.add_node(child, parent_key=ptr)
                        queue.push((child, child_ptr))
        
        return tree
    
    def prune_tree(self, tree):
        return tree