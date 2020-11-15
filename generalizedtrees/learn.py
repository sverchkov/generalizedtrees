# Entrypoint for API v 1.0
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

from generalizedtrees.split import SplitCandidateGeneratorLC, SplitScoreLC
from generalizedtrees.stop import GlobalStopLC, LocalStopLC, NeverStopLC
from generalizedtrees.queues import CanPushPop
from typing import Callable
import numpy as np
from pandas import DataFrame

from generalizedtrees.tree import Tree, tree_to_str
from generalizedtrees.predict import PredictorLC
from generalizedtrees.givens import GivensLC
from generalizedtrees.grow import GreedyBuilderLC, NodeBuilderLC

class GreedyTreeLearner:

    # Internals
    tree: Tree

    # Learner components
    givens: GivensLC
    builder: GreedyBuilderLC = GreedyBuilderLC()
    predictor: PredictorLC
    
    # TODO?
    #def __init__(self, *args, **kwargs):
        # TODO: Validation
        # TODO: Set components from input

    # Component setters for components that aren't direct attributes

    @property
    def local_stop(self) -> LocalStopLC:
        return self.builder.local_stop
    
    @local_stop.setter
    def local_stop(self, value: LocalStopLC) -> None:
        self.builder.local_stop = value
    
    @property
    def global_stop(self) -> GlobalStopLC:
        return self.builder.global_stop
    
    @global_stop.setter
    def global_stop(self, value: GlobalStopLC):
        self.builder.global_stop = value

    @property
    def split_score(self) -> SplitScoreLC:
        return self.builder.splitter.split_scorer
    
    @split_score.setter
    def split_score(self, value: SplitScoreLC):
        self.builder.splitter.split_scorer = value

    @property
    def split_generator(self) -> SplitCandidateGeneratorLC:
        return self.builder.splitter.split_generator
    
    @split_generator.setter
    def split_generator(self, value: SplitCandidateGeneratorLC):
        self.builder.splitter.split_generator = value
    
    @property
    def node_builder(self) -> NodeBuilderLC:
        return self.builder.node_builder
    
    @node_builder.setter
    def node_builder(self, value: NodeBuilderLC):
        self.builder.node_builder = value

    def set_queue(self, queue = Callable[..., CanPushPop]):
        self.builder.new_queue = queue

    # Setting individual components after initialization shoud use python property/attribute setting syntax
    queue = property(None, set_queue)

    # Read-only properties
    @property
    def target_names(self):
        return self.givens.target_names
    
    @property
    def feature_names(self):
        return self.givens.feature_names

    def fit(self, *args, **kwargs):
        
        # TODO: Check for things in kwargs that set other component parameters

        # TODO: Check that required things are set

        # Process givens
        self.givens.process(*args, **kwargs)
        
        # Set components
        self.predictor.initialize(self.givens)
        self.builder.node_builder.initialize(self.givens)
        self.split_generator.initialize(self.givens)

        # Build tree
        self.tree = self.builder.build_tree()

        # Prune tree
        self.tree = self.builder.prune_tree(self.tree)

        return self
    
    def predict(self, data):

        self._checks_to_predict()
        data_matrix = self._validate_data(data)

        return self.predictor.predict(self.tree, data_matrix)

    def predict_proba(self, data):

        self._checks_to_predict()
        data_matrix = self._validate_data(data)

        return self.predictor.predict_proba(self.tree, data_matrix)

    def _validate_data(self, data) -> np.ndarray:

        # TODO: more validation

        if isinstance(data, DataFrame):
            data = data.to_numpy()
        
        return data

    def _checks_to_predict(self) -> None:
        self._check_tree()
        self._check_predictor()

    def _check_predictor(self) -> None:
        try:
            self.predictor
        except AttributeError:
            raise ValueError('No predictor present.')

    def _check_tree(self) -> None:
        try:
            self.tree
        except AttributeError:
            raise ValueError('Tried to predict before learning the tree.')
    
    def show_tree(self):

        def show_node(node_obj):
            if node_obj.split is None:
                if node_obj.model is None:
                    logger.critical('Malformed node encountered in tree printing.')
                    return "Malformed node"
                else:
                    return str(node_obj.model)
            else:
                return str(node_obj.split)

        return tree_to_str(self.tree, show_node)