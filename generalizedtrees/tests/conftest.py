# Definitions for shared test fixtures
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov


import pytest

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from collections import namedtuple

# Housekeeping
np.random.seed(20200617)

Dataset = namedtuple(
    'Dataset',
    ['x_train', 'x_test', 'y_train', 'y_test', 'feature_names', 'target_names'])

@pytest.fixture(scope="module")
def breast_cancer_data():

    # Load data
    bc = load_breast_cancer()

    x_train, x_test, y_train, y_test = train_test_split(bc['data'], bc['target'])

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

@pytest.fixture(scope="module")
def bc_lowD_data(breast_cancer_data):

    Dataset = namedtuple(
        'Dataset',
        ['x_train', 'x_test', 'y_train', 'y_test', 'feature_names', 'target_names'])

    return Dataset(
        x_train=breast_cancer_data.x_train[:, 0:3],
        x_test=breast_cancer_data.x_test[:, 0:3],
        y_train=breast_cancer_data.y_train,
        y_test=breast_cancer_data.y_test,
        feature_names=breast_cancer_data.feature_names[0:3],
        target_names=breast_cancer_data.target_names
    )

@pytest.fixture(scope="module")
def bc_lowD_rf_model(bc_lowD_data):

    # Learn 'black-box' model
    rf = RandomForestClassifier()
    rf.fit(bc_lowD_data.x_train, bc_lowD_data.y_train)

    return rf

@pytest.fixture(scope='module')
def small_weights_data():

    x_train = np.random.randint(2, size=(500, 10)).astype(bool)
    x_test = np.random.randint(2, size=(100, 10)).astype(bool)
    y_train = x_train[:, 0:4].sum(axis=1) > x_train[:, 6:9].sum(axis=1)
    y_test = x_test[:, 0:4].sum(axis=1) > x_test[:, 6:9].sum(axis=1)
    
    return Dataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=[f'feature {i}' for i in range(10)],
        target_names=['left', 'right']
    )

@pytest.fixture(scope='module')
def small_weights_model(small_weights_data):

    # Learn model
    lr = LogisticRegression()
    lr.fit(small_weights_data.x_train, small_weights_data.y_train)

    return lr