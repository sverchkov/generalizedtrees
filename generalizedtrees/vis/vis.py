# Visualization functions for our trees
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

import pkgutil
import json
from logging import getLogger
from generalizedtrees.queues import Stack

logger = getLogger()

TEMPLATE_STR = '/***DATA***/'

def _load_dt_template():
    return pkgutil.get_data('generalizedtrees.vis', 'decision_tree.html')

def explanation_to_html(explanation, out_file):

    json_str = explanation_to_JSON(explanation)
    # We anticipate possibly using different templates depending on model type
    template = _load_dt_template()

    if hasattr(out_file, 'write'):
        _insert_JSON(json_str, template, out_file)
    else:
        with open(out_file) as f:
            _insert_JSON(json_str, template, f)

def _insert_JSON(json_str, template, out_file):
    
    out_file.write(template.replace(TEMPLATE_STR, 'data = ' + json_str + ';'))

def explanation_to_JSON(explanation):

    root = dict()

    stack = Stack()
    stack.push((root, explanation.tree.node('root')))

    while (stack):

        out_node, in_node = stack.pop()

        # Record split
        if hasattr(in_node.item, 'split') and in_node.item.split is not None:
            out_node['split'] = str(in_node.item.split)
        
        # Record children
        if not in_node.is_leaf:
            out_node['children'] = [{'branch': str(c.item.local_constraint)} for c in in_node]
            for pair in zip(out_node['children'], in_node):
                stack.push(pair)
        
        # Node-model specific conversions
        # should it be the node class' responsibility to implement these?
        if hasattr(in_node.item, 'probabilities'):
            out_node['probabilities'] = dict(in_node.item.probabilities)
        
        if hasattr(in_node.item, 'model'):
            model = in_node.item.model
            # Monkey check for LR/linear model
            # TODO: Handle case when intercept+coef are multi-d
            if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                if hasattr(model.intercept_, 'shape'):
                    intercept = model.intercept_.flat[0]
                else:
                    intercept = model.intercept_
                out_node['logistic_model'] = {'intercept': intercept}
                try:
                    coefficients = {key: value
                        for (key, value) in zip(explanation.feature_names, model.coef_.flat)
                        if abs(value) > 0}
                except:
                    logger.critical(
                        'Could not convert logistic model to JSON:'
                        f'\ncoef_: {model.coef_}')
                    raise
                out_node['logistic_model'].update(coefficients)
    
    return json.dumps(root)
