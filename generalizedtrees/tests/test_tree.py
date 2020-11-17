# Tests for our tree data structure
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

def test_tree_building(caplog):
    from logging import getLogger
    from generalizedtrees.tree import Tree, tree_to_str

    logger = getLogger()

    t = Tree(['A', ['B', ['H'], ['I']], ['C', 'E', 'F', 'G'], 'D'])
    
    assert(len(t) == 9)
    assert(t.depth == 2)

    logger.info(f'Tree as list: {list(t)}')

    logger.info(f'Pretty-printed tree:\n{tree_to_str(t)}')
