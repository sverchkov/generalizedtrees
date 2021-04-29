.. image:: https://github.com/Craven-Biostat-Lab/generalizedtrees/actions/workflows/python-pytest.yml/badge.svg
    :alt: Build Status
    :target: https://github.com/Craven-Biostat-Lab/generalizedtrees/actions/workflows/python-pytest.yml

.. image:: https://codecov.io/gh/Craven-Biostat-Lab/generalizedtrees/branch/master/graph/badge.svg
    :alt: codecov
    :target: https://codecov.io/gh/Craven-Biostat-Lab/generalizedtrees

.. image:: https://img.shields.io/pypi/pyversions/generalizedtrees.svg
    :alt: Python Versions
    :target: https://pypi.python.org/pypi/generalizedtrees

.. image:: https://badge.fury.io/py/generalizedtrees.svg
    :alt: PyPI version
    :target: https://badge.fury.io/py/generalizedtrees

.. image:: https://img.shields.io/pypi/l/generalizedtrees.svg
    :alt: License
    :target: https://pypi.python.org/pypi/generalizedtrees

================
generalizedtrees
================

Library for tree models: decision trees, model trees, mimic models, etc.

Installation
============

The latest stable version is on PyPI.
Install it with::

    pip install generalizedtrees

Dependencies
============
* Python (>=3.8)
* scikit-learn (>=0.23.2)
* numpy (>=1.19.1)
* scipy (>=1.5.2)
* pandas (>=1.1.0)

See requirements.txt for recommended dependencies (usually newest versions of all packages).

Development
===========

We use the `GitHub issue tracker`_ for bugs and feature requests.

Building
--------

We use the standard Python process for building the package.
Run::

    python setup.py build

to locally build the package, and::

    python setup.py install

to install the locally built package.

Testing
-------

We use pytest_ for testing.
Run::

    pytest

on the command line to run all tests.

License
=======

Licensed under the BSD 3-Clause License Copyright (c) 2020, Yuriy Sverchkov

See LICENSE



.. _`GitHub issue tracker`: https://github.com/Craven-Biostat-Lab/generalizedtrees/issues
.. _pytest: https://docs.pytest.org/en/latest/
