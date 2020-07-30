# Feature type specification (classes and utilities)
#
# Copyright 2019 Yuriy Sverchkov
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


from enum import Flag, auto
import pandas as pd #?
import logging

logger = logging.getLogger()


class FeatureSpec(Flag):
    ORDERED = auto()
    DISCRETE = auto()
    ORDINAL = ORDERED | DISCRETE
    CONTINUOUS = ORDERED # And not discrete


def infer_feature_spec(data_matrix):

    if isinstance(data_matrix, pd.DataFrame):

        # Infer based on column types
        raise NotImplementedError('Coming soon!')
    
    else:
        raise NotImplementedError('Feature type inference only implemented for pandas DataFrames')