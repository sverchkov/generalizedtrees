# Text-based visualization

from logging import getLogger
from generalizedtrees.constraints import SimpleConstraint

from generalizedtrees.tree import Tree, tree_to_str

logger = getLogger()

class TreePrinter:

    def __init__(self, feature_names = None):
        self.feature_names = feature_names

    def show(self, tree: Tree):

        return tree_to_str(tree, self.show_node)
    
    def show_node(self, node_obj):
        lc = node_obj.local_constraint
        if lc:
            if self.feature_names is not None and isinstance(lc, SimpleConstraint):
                lc_str = f'{self.feature_names[lc.feature]} {lc.operator.value} {lc.value}'
            else:
                lc_str = str(lc)
        else:
            lc_str = 'Root'

        if node_obj.split is None:
            if node_obj.model is None:
                logger.critical('Malformed node encountered in tree printing.')
                return "Malformed node"
            else:
                return f'{lc_str}: {str(node_obj.model)}'
        else:
            return lc_str
