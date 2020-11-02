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

import numpy as np
from pandas import DataFrame

from generalizedtrees.tree import Tree
from generalizedtrees.predict import PredictorLC
from generalizedtrees.givens import GivensLC

class GreedyTreeLearner:

    # Internals
    tree: Tree
    target_names: np.ndarray

    # Learner components
    givens: GivensLC
    predictor: PredictorLC

    # TODO
    #def __init__(self, *args, **kwargs):
        # TODO: Validation
        # TODO: Set components from input

    # TODO: Individual component setters

    def fit(self, *args, **kwargs):
        
        # TODO: Check for things in kwargs that set other component parameters

        # Process givens
        self.givens.process(*args, **kwargs)
        
        self.predictor.set_target_names(self.givens.target_names)

        # Build tree
        # Prune tree

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