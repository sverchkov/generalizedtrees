# Abstract tree learner class
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

from abc import ABC, abstractmethod

class TreeLearner(ABC):

    def _build(self):

        root = self.new_node()
        self.tree = root.plant_tree()

        queue = self.new_queue()
        queue.push(root)

        while queue and not self.global_stop():

            node = queue.pop()

            for branch in self.construct_split(node):

                child = self.new_node(branch, node)
                node.add_child(child)

                if not self.local_stop(child):
                    queue.push(child)
    
    @abstractmethod
    def new_node(self, branch = None, parent = None):
        pass

    @abstractmethod
    def new_queue(self):
        pass

    @abstractmethod
    def construct_split(self, node):
        pass

    @abstractmethod
    def global_stop(self):
        pass

    @abstractmethod
    def local_stop(self, node):
        pass