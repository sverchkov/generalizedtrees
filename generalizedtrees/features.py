# Feature type specification (classes and utilities)
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov


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