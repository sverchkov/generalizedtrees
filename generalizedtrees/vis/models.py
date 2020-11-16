# Visualization functions for models in trees
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

from typing import Union

from numpy import generic, ndarray, concatenate, array
from logging import getLogger

from generalizedtrees.leaves import ConstantEstimator, SKProbaClassifier, BinSKProbaEstimator

logger = getLogger()

def model_to_simplified(model, explanation):
    """
    Main entry point for converting local models to a simplified object.
    
    A simplified object is a structure made up of dicts, lists, and primitives.
    Dispatches helpers based on object inspection.
    """
    if isinstance(model, ConstantEstimator):
        return _constant_estimator_to_simplified(model, explanation)
    
    if isinstance(model, (SKProbaClassifier, BinSKProbaEstimator)):
        if hasattr(model.classifier, 'intercept_') and hasattr(model.classifier, 'coef_'):
            return _skl_linear_estimator_to_simplified(model, explanation)
    
    # Fall through default
    return str(model)

def _ensure_native(scalar):
    if isinstance(scalar, ndarray):
        assert len(scalar) == 1
        scalar = scalar[0]
    if isinstance(scalar, generic): return scalar.item()
    return scalar

def _constant_estimator_to_simplified(model: ConstantEstimator, explanation):

    # Check case of n-1-d estimator for n-ary classification
    est_vector = model.est_vector
    try:
        if len(est_vector) < len(explanation.target_names):
            est_vector = concatenate([array([1-est_vector.sum()]), est_vector], axis=0)
    except IndexError:
        # est_vector must be scalar?
        est_vector = [1-est_vector, est_vector] # Don't have to use numpy arrays here

    return {'estimate': [
        {
            'label_id': int(i),
            'label': _ensure_native(explanation.target_names[i]),
            'value': _ensure_native(est_vector[i])}
        for i in range(len(explanation.target_names))]}

def _skl_linear_estimator_to_simplified(
    model: Union[SKProbaClassifier, BinSKProbaEstimator],
    explanation,
    epsilon=1E-6
    ):

    cfer = model.classifier

    try:
        coefficients = (
            [{'label': 'intercept', 'value': _ensure_native(cfer.intercept_)}] +
            [
                {
                    'label': _ensure_native(explanation.feature_names[i]),
                    'value': _ensure_native(cfer.coef_.flat[i]),
                    'feature_id': int(i)
                }
                for i in range(len(cfer.coef_))
                if abs(cfer.coef_.flat[i]) > epsilon]
        )
    except:
        logger.critical(
            'Could not convert (generalized) linear model to JSON:'
            f'\nmodel: {str(cfer)}'
            f'\nintercept: {cfer.intercept_}'
            f'\ncoef_: {cfer.coef_}')
        raise
    
    return {'coefficients': coefficients}