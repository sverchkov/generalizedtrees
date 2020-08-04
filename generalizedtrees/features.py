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
import pandas as pd #?
import logging

logger = logging.getLogger()


class FeatureSpec(Flag):
    ORDERED = auto()
    DISCRETE = auto()
    ORDINAL = ORDERED | DISCRETE
    CONTINUOUS = ORDERED # And not discrete


def infer_feature_spec_of_dtype(d_type) -> FeatureSpec:
    """
    Maps numpy dtypes to FeatureSpecs.

    Makes use of dtype.kind attribute (see numpy documentation)
    Currently, classifies datatypes as follows:
    Continuous (ordered and not discrete):
        i (signed integer) *
        u (unsigned integer) *
        f (floating point)
        m (timedelta)
        M (datetime)
    The remaining datatypes are classified as discrete (and unordered):
        b (boolean) *
        c (complex floating-point) **
        O (object)
        S (byte-)string
        U Unicode
        V void
    
    * Booleans and integers are technically discrete and ordered. Current
    code isn't set up to handle that, so we classify as above.
    ** Complex floating-point is technically non-ordered non-discrete, but
    we don't anticipate handling complex numbers in the forseeable future.
    """
    if d_type.kind in 'iufmM':
        return FeatureSpec.CONTINUOUS
    else:
        return FeatureSpec.DISCRETE


def infer_feature_spec(data_matrix) -> Tuple(FeatureSpec):

    if isinstance(data_matrix, pd.DataFrame):

        return tuple(data_matrix.dtypes.map(infer_feature_spec_of_dtype))
    
    else:
        raise NotImplementedError('Feature type inference only implemented for pandas DataFrames')