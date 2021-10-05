# Text-based visualization

from logging import getLogger

from generalizedtrees.tree import Tree, tree_to_str

logger = getLogger()

class TreePrinter:

    def show(self, tree: Tree):

        def show_node(node_obj):
            lc = node_obj.local_constraint
            lc_str = str(lc) if lc else 'Root'
            if node_obj.split is None:
                if node_obj.model is None:
                    logger.critical('Malformed node encountered in tree printing.')
                    return "Malformed node"
                else:
                    return f'{lc_str}: {str(node_obj.model)}'
            else:
                return lc_str #f'{lc_str}: {str(node_obj.split)}'

        return tree_to_str(tree, show_node)