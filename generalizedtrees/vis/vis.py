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
from generalizedtrees.queues import Stack

TEMPLATE_STR = '/***DATA***/'
DT_TEMPLATE = pkgutil.get_data('generalizedtrees.vis', 'decision_tree.html')

def explanation_to_html(explanation, out_file):

    json_str = explanation_to_JSON(explanation)

    with open(DT_TEMPLATE) as in_file:
        if hasattr(out_file, 'write'):
            _insert_JSON(json_str, in_file, out_file)
        else:
            with open(out_file) as f:
                _insert_JSON(json_str, in_file, f)

def _insert_JSON(json_str, in_file, out_file):
    
    out_file.write(in_file.read().replace(TEMPLATE_STR, 'data = ' + json_str + ';'))

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
            out_node['children'] = [{'branch': c.item.local_constraint} for c in out_node]
            for pair in zip(out_node['children'], in_node):
                stack.push(pair)
        
        # Node-model specific conversions
        # should it be the node class' responsibility to implement these?
        if hasattr(in_node.item, 'probabilities'):
            out_node['probabilities'] = dict(in_node.item.probabilities)
        
        if hasattr(in_node.item, 'model'):
            model = in_node.item.model
            # Monkey check for LR/linear model
            if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                out_node['logistic_model'] = {'intercept': model.intercept_}
                out_node['logistic_model'].update(dict(model.coef_))
    
    return json.dumps(root)
