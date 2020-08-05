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
from typing import Tuple
import pandas as pd
import logging

logger = logging.getLogger()

category_dtype = pd.CategoricalDtype()

class FeatureSpec(Flag):
    # Base flags:
    ORDERED = auto()
    DISCRETE = auto()
    # Derived flags:
    ORDINAL = ORDERED | DISCRETE
    CONTINUOUS = ORDERED # And not discrete
    UNCOUNTABLE = 0 # Neither ordered nor discrete


def infer_feature_spec_of_dtype(d_type) -> FeatureSpec:
    """
    Maps numpy/pandas dtypes to FeatureSpecs.

    'category' dtypes are discrete, and treated according to their 'ordered' attribute.

    Makes use of dtype.kind attribute (see numpy documentation)
    Currently, classifies datatypes as follows:
    Continuous (ordered and not discrete):
        f (floating point)
        m (timedelta)
        M (datetime)
    Ordinal (ordered and discrete)
        i (signed integer)
        u (unsigned integer)
        b (boolean)
    Neither ordered nor discrete:
        c (complex floating-point)
    Remaining types are discrete (and unordered):
        O (object)
        S (byte-)string
        U Unicode
        V void
    """
    if category_dtype == d_type:
        if d_type.ordered:
            return FeatureSpec.ORDINAL
        else:
            return FeatureSpec.DISCRETE
    elif d_type.kind in 'fmM':
        return FeatureSpec.CONTINUOUS
    elif d_type.kind in 'iub':
        return FeatureSpec.ORDINAL
    elif d_type.kind in 'c':
        return FeatureSpec.UNCOUNTABLE
    else:
        return FeatureSpec.DISCRETE


def infer_feature_spec(data_matrix) -> Tuple[FeatureSpec]:

    if isinstance(data_matrix, pd.DataFrame):

        return tuple(data_matrix.dtypes.map(infer_feature_spec_of_dtype))
    
    else:
        raise NotImplementedError('Feature type inference only implemented for pandas DataFrames')