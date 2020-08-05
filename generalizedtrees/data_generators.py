# Data generators (used in oracle-based explainers)
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

from generalizedtrees.features import FeatureSpec
from collections import namedtuple
import numpy as np
import pandas as pd


Generator = namedtuple("Generator", ["generate", "training_data"])

def trepan_generator(tree_model, training_data: pd.DataFrame):
    """
    The data generation scheme used in trepan
    """

    d = training_data.shape[1]
    cols = training_data.columns

    # The Trepan generator independently generates the individual feature values.
    feature_generators = [
        _feature_generator(training_data.iloc[:,i], tree_model.feature_spec[i])
        for i in range(d)]

    return Generator(
        generate = lambda: pd.Series([f() for f in feature_generators], index=cols),
        training_data = training_data)

def _feature_generator(data_vector, feature: FeatureSpec, rng=np.random.default_rng()):

        n = len(data_vector)

        if feature is FeatureSpec.CONTINUOUS:
            # Sample from a KDE.
            # We use Generator and not RandomState but KDE implementations use RandomState
            # so it's more reliable to just implement the sampling here. 
            return lambda: rng.normal(
                loc = rng.choice(data_vector, size=1),
                scale = 1/np.sqrt(n),
                size = 1)[0]

        elif feature & FeatureSpec.DISCRETE:
            # Sample from the empirical distribution
            values, counts = np.unique(data_vector, return_counts=True)
            return lambda: rng.choice(values, p=counts/n)
        
        else:
            raise ValueError(f"I don't know how to handle feature spec {feature}")