============
Introduction
============

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

Usage
=====

Most workflows start by creating a `GreedyTreeLearner` object, fitting it to data (and, if
applicable, an oracle), and inspecting or using the result.