# Visualization functions for our trees
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

from string import Template
import pkgutil
import json
from logging import getLogger
from generalizedtrees.queues import Stack
from generalizedtrees.vis.models import model_to_simplified
import generalizedtrees.constraints as cons
from generalizedtrees.vis.util import _ensure_native

logger = getLogger()

def _load_html_template(template = 'template.html'):
    return pkgutil.get_data('generalizedtrees.vis', template).decode('utf-8')


def _load_dt_js():
    return pkgutil.get_data('generalizedtrees.vis', 'decision_tree.js').decode('utf-8')


def explanation_to_html(explanation, out_file, feature_annotations = None):

    # We anticipate possibly using different templates depending on model type
    html_str = Template(_load_html_template()).substitute({
        'data': explanation_to_JSON(explanation, feature_annotations),
        'artist': _load_dt_js()
    })

    if hasattr(out_file, 'write'):
        out_file.write(html_str)
    else:
        with open(out_file, 'wt') as f:
            f.write(html_str)


def _get_constraint_type_as_html_string(constraint):
    if isinstance(constraint, cons.SimpleConstraint):
        return f'{constraint.operator.value} {constraint.value:.5g}'
    # Fall through catch-all
    return str(constraint)


def explanation_to_JSON(explanation, feature_annotations = None):
    simplified = explanation_to_simplified(explanation, feature_annotations)
    try:
        return json.dumps(simplified)
    except:
        logger.debug(f'Simplified model: {simplified}')
        raise


def explanation_to_simplified(explanation, feature_annotations = None):

    root = dict()

    stack = Stack()
    stack.push((root, explanation.tree.node('root')))

    while stack:

        out_node, in_node = stack.pop()

        # Record training set counts (and target distributions)
        if hasattr(in_node.item, 'y'):
            if hasattr(in_node.item, 'n_training'):
                y_train = in_node.item.y[0:in_node.item.n_training]
                y_gen = in_node.item.y[in_node.item.n_training:]

                out_node['generated_samples'] = [
                    {'label': str(k), 'count': int(v)} for v, k in
                    zip(y_gen.sum(axis=0), explanation.target_names)]
            else:
                y_train = in_node.item.y

            out_node['training_samples'] = [
                {'label':str(k), 'count': int(v)} for v, k in
                zip(y_train.sum(axis=0), explanation.target_names)]


        # Record split
        if hasattr(in_node.item, 'split') and in_node.item.split is not None:
            out_node['split'] = str(in_node.item.split)

            if feature_annotations is not None:
                try:
                    f = in_node.item.split.feature
                    annotation = feature_annotations.loc[f]
                    out_node['feature_annotation'] = [{
                        'annotation': 'feature id',
                        'value': _ensure_native(f)}]
                    out_node['feature_annotation'].extend([
                        {'annotation': str(i), 'value': _ensure_native(v)}
                        for i, v in annotation.iteritems()])
                
                except Exception as e:
                    logger.warning(
                        f"Could not bind feature annotation to {in_node.item.split}",
                        exc_info=e)
        
        # Record children
        if not in_node.is_leaf:
            out_node['children'] = [
                {'branch': _get_constraint_type_as_html_string(c.item.local_constraint)}
                for c in in_node]
            for pair in zip(out_node['children'], in_node):
                stack.push(pair)
        
        # Node-model specific conversions should be implemented elsewhere.
        if hasattr(in_node.item, 'model') and in_node.item.model is not None:
            # Currently a stump
            out_node['model'] = model_to_simplified(in_node.item.model, explanation)
    
    return root
