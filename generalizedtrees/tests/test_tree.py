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

from generalizedtrees.tree import Tree, tree_to_str

def test_tree_building():

    t = Tree("A")
    r = t.root
    r.add_children(["B", "C", "D"])
    r[1].add_children(["E", "F", "G"])
    r[0].add_children(["H", "I"])

    assert(len(t) == t.size)
    assert(t.size == 9)

    print(t[:])

    print(tree_to_str(t))