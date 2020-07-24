# Implementations of stopping criteria
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


## Local stopping criteria

def perfect_classification(self, node):
    return node.probabilities.max() >= 1

def node_depth(tree_model, node):
    return node.depth >= tree_model.max_depth

## Global stopping criteria

def never(tree_model):
    return False

def tree_size_limit(self):
    return self.tree.size >= self.max_tree_size
