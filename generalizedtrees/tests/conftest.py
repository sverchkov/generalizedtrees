# Definitions for shared test fixtures
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov


import pytest

from numpy.random import seed
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from collections import namedtuple

# Housekeeping
seed(20200617)

@pytest.fixture(scope="module")
def breast_cancer_data():

    # Load data
    bc = load_breast_cancer()

    x_train, x_test, y_train, y_test = train_test_split(bc['data'], bc['target'])

    Dataset = namedtuple(
        'Dataset',
        ['x_train', 'x_test', 'y_train', 'y_test', 'feature_names', 'target_names'])

    return Dataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=bc['feature_names'],
        target_names=bc['target_names'])

@pytest.fixture(scope="module")
def breast_cancer_data_pandas():

    # Load data
    bc = load_breast_cancer(as_frame=True)

    x_train, x_test, y_train, y_test = train_test_split(bc['data'], bc['target'])

    return Bunch(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        target_names=bc['target_names'])

@pytest.fixture(scope="module")
def breast_cancer_rf_model(breast_cancer_data):

    # Learn 'black-box' model
    rf = RandomForestClassifier()
    rf.fit(breast_cancer_data.x_train, breast_cancer_data.y_train)

    return rf