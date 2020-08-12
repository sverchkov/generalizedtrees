# Utility for composing tree learners out of building blocks
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

from typing import Type, Any
from dataclasses import make_dataclass
from generalizedtrees.base import GreedyTreeBuilder
from generalizedtrees.classification import TreeClassifierMixin
from generalizedtrees.tree import tree_to_str
from logging import getLogger

logger = getLogger()

# Show tree
def show_tree(model_obj):
    return tree_to_str(model_obj.tree)

# Composing a split constructor

def compose_sample_based_split_constructor(
    split_candidate_generator,
    split_score):
    """
    Construct splits based on sample data associated with a tree node.

    Node needs to have properties `data` and `targets`
    """
    def construct_split(tree_model, node):
        data = node.data
        targets = node.targets
        feature_spec = tree_model.feature_spec

        split_candidates = split_candidate_generator(feature_spec, data, targets)
        best_split = None
        best_split_score = 0
        for split in split_candidates:
            new_score = split_score(split, data, targets)
            if new_score > best_split_score:
                best_split_score = new_score
                best_split = split
        
        logger.debug(f'Best split "{best_split}" with score {best_split_score}')

        return best_split
    
    return construct_split

# Composing a classification tree learner

def greedy_classification_tree_learner(
    name: str,
    parameters, # As in dataclass
    fitter, # Fit function
    node_building: Type[Any], # Mixin class
    split_candidate_generator, # Function (feature spec, data, targets)
    split_score, # Function (split, data, targets)
    queue: Type[Any], # Queue class
    global_stop, # function of model
    local_stop, # function of model, node
    data_generator = None # Function, optional. Takes training data and outputs f(constraints, n) -> data
    ):

    bases = (node_building, TreeClassifierMixin, GreedyTreeBuilder)

    members = dict(
        fit=fitter,
        new_queue=queue,
        construct_split=compose_sample_based_split_constructor(
            split_candidate_generator,
            split_score),
        global_stop=global_stop,
        local_stop=local_stop,
        show_tree=show_tree
    )

    if data_generator is not None:
        members.update(dict(new_generator=data_generator))

    #C = type('C', (TreeBuilder, TreeClassifierMixin, node_building), members)

    return make_dataclass(name, fields=parameters, bases=bases, namespace=members)