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

def test_tree_building(caplog):
    from logging import getLogger
    from generalizedtrees.tree import Tree, tree_to_str

    logger = getLogger()

    t = Tree(['A', ['B', ['H'], ['I']], ['C', 'E', 'F', 'G'], 'D'])
    
    assert(len(t) == 9)
    assert(t.depth == 2)

    logger.info(f'Tree as list: {list(t)}')

    logger.info(f'Pretty-printed tree:\n{tree_to_str(t)}')
