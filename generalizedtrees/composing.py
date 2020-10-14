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

from typing import Type, Any, Callable
from dataclasses import make_dataclass
from generalizedtrees.base import GreedyTreeBuilder, CanPushPop
from generalizedtrees.predict import TreeClassifierMixin
from generalizedtrees.tree import tree_to_str
from logging import getLogger

logger = getLogger()

# Show tree
def show_node(node_obj):
    if node_obj.split is None:
        if node_obj.model is None:
            logger.critical('Malformed node encountered in tree printing.')
            return "Malformed node"
        else:
            return str(node_obj.model)
    else:
        return str(node_obj.split)

def show_tree(model_obj):
    return tree_to_str(model_obj.tree, show_node)

# Composing a split constructor

def compose_sample_based_split_constructor(
    split_candidate_generator,
    split_score,
    use_proba=False):
    """
    Construct splits based on sample data associated with a tree node.

    Node needs to have properties `data` and `targets` (or `target_proba` if
    use_proba is true)
    """
    def construct_split(tree_model, node):
        data = node.data
        y = node.target_proba if use_proba else node.targets
        feature_spec = tree_model.feature_spec

        split_candidates = split_candidate_generator(feature_spec, data, y)
        best_split = None
        best_split_score = 0
        for split in split_candidates:
            new_score = split_score(split, data, y)
            if new_score > best_split_score:
                best_split_score = new_score
                best_split = split
        
        logger.debug(f'Best split "{best_split}" with score {best_split_score}')

        return best_split
    
    return construct_split

def compose_split_constructor_mk2(split_candidate_generator, split_score, *ignoreArgs):
    """
    Create a split constructor that finds an optimal split for a node. Mk2.

    :param split_candidate_generator: Function to generate split candidates, should have
    a signature of {feature_spec, data, y -> iterable(split)}
    :param split_score: Split scoring function, should have a signature of {node, split -> float}
    """
    def construct_split(tree_model, node):
        data = node.data
        y = node.target_proba # Old approach, need to change to some variant of 'estimate'
        feature_spec = tree_model.feature_spec

        split_candidates = split_candidate_generator(feature_spec, data, y)
        best_split = None
        best_split_score = 0
        for split in split_candidates:
            new_score = split_score(node, split)
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
    fitter: Callable, # Fit function
    node_building: Type[Any], # Mixin class
    split_candidate_generator: Callable, # Function (feature spec, data, targets)
    split_score: Callable, # Function (split, data, targets)
    queue: Type[CanPushPop], # Queue class
    global_stop: Callable, # function of model
    local_stop: Callable, # function of model, node
    data_generator = None, # Function, optional. Takes training data and outputs f(constraints, n) -> data
    **kwargs):

    bases = (node_building, TreeClassifierMixin, GreedyTreeBuilder)

    use_proba = kwargs.get('use_proba', False)
    splits_composer = kwargs.get('splits_composer', compose_sample_based_split_constructor)

    members = dict(
        fit=fitter,
        new_queue=queue,
        construct_split=splits_composer(
            split_candidate_generator,
            split_score,
            use_proba),
        global_stop=global_stop,
        local_stop=local_stop,
        show_tree=show_tree
    )

    if data_generator is not None:
        members.update(dict(new_generator=data_generator))

    return make_dataclass(name, fields=parameters, bases=bases, namespace=members)