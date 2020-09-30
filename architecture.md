# Generalized tree learning

The main purpose of this package is to serve as a library of decision tree classifiers and
regressors, both learned from supervised data and learned to explain other classifiers.

What we aim to do here is create a general framework and a collection of ingredients that go
into learning a decision tree.
These ingredients can then be combined to create an object that performs the learning/explanation task.

The following outlines how we abstract the task in general terms.

## Anatomy of a decision tree

A decision tree is a tree the internal (non-leaf) nodes of which correspond to splits and the leaves
of which correspond to predictors.

### Splits

A split is an object that given a data matrix of $n$ instances of $m$ features, maps each instance to
one of $k$ branches.
Traditionally, each branch corresponds to a relativelty simple constraint on a feature, but one of our goals is to be able to support complex constraints.

### Leaf predictors

Each leaf node must be able to, given $n$ instances of $m$ features, produce a target output for each instance.
In the general case, the target output is a vector of size $t$.

## Evaluating a decision tree

For a data matrix of $n$ instances of $m$ features, a decision tree produces a target matrix of $n$ instances by $t$ target components. 

## Constructing a decision path

*Status: Not implemented*

In some applications it may be of interest to query the decision path for a particular instance.

## Learning a decision tree:

Abstractly, to learn a decision tree one needs to construct a tree object given data (or in the case of an explanation, a model) to fit.
This entails specifying a fitter

* Supervised learning fitter: takes data and targets
* Type 1 explanation fitter: takes data, a predictor, and a recipe for generating unlabeled data based on given data
* Type 2 explanation fitter: takes a predictor and an unlabeled data generator. *Status: Not implemented*

The fitter must then call a builder that builds the tree given the provided information.
The most common build strategy is a greedy building of the tree, followed (optionally) by pruning of the tree.

### Greedy tree building

The greedy tree building process is the following:

1. Initialize a root node
2. Push the node into a queue (or any sequential access data structure such as a stack or heap).
3. While the queue is not empty and global stopping criteria aren't met: 
    1. Pop a node from the queue
    2. Find the locally best split at this node
    3. If there is no good split or if local stopping criteria are met, **make this node a leaf**
    4. Else:
        1. Set the identified split to be the split at the node.
        2. Initialize children based on the split (one for each split branch)
        3. Push the children into the queue.

The elements that vary in the building process from algorithm to algorithm are:

* The sequential access data structure
* The algorithm for finding the locally best split
* The algorithm for building a leaf node
* The local stopping criteria
* The global stopping criteria

### Finding the locally best split

**TODO**

# Recipes

For convenience we provide several 'recipes' for standard learners, such as a classic decision tree learner and Trepan.
These are found in `generalizedtees.recipes` and the source code therein can be used as an example for composing learners.

# [Scikit Learn] compatibility

Initially we hoped to make the built estimators pass sklearn's `sklearn.utils.estimator_checks.check_estimator` check, but this proves challenging with composed models.
Instead, we aim for a consistent API in terms of method names (e.g. `fit`, `predict`, `predict_proba`) which should suffice in many cases for working alongside sklearn code.

*Future consideration:* We will revisit the feasibility of inheriting from the
[sklearn.base.BaseEstimator] base class and the [sklearn.base.ClassifierMixin] ([sklearn.base.RegressorMixin]) mixin.

## Scikit-learn rules

For completeness we include notes on what one would need to do to pass sklearn checks (not currently the case for our estimators).

To be able to pass the scikit-learn checks an estimator must:

 * All arguments of `__init__` must have a default value
 * All arguments of `__init__` must correspond to class attributes.
 * `__init__` cannot set attributes that do not correspond to its arguments.
 * The second parameter of `fit` must be `y` or `Y`.
 * See the following subsection for a description of what is considered to be part of the scikit-learn interface

### The scikit-learn interface

| Method | Implemented in |
| --- | --- |
| `predict(data)` | |
| `fit(data,[y]) | |
| `score(data, y)` | `BaseEstimator` |
| `get_params` and `set_params` | `BaseEstimator` |

### Other scikit-learn checks

Particularly, with `sklearn.utils.estimator_checks.check_estimator`.
Here we document some of the less-obvious checks it performs:

 * Passing an object to the `y` argument of `fit` should raise an exception.
 Use `sklearn.utils.multiclass.check_classification_targets` to ensure it.
 * `fit` returns `self`
 * Predict raises an error for infinite and NaN inputs. (I prefer a more robust implementation of decision trees that can handle infinite and missing values).
 * Classifiers must have a `classes_` element
 * Trying to predict without fitting must raise an error containing the word "fit"

---
Licensed under the Apache License, Version 2.0, Copyright 2020 Yuriy Sverchkov

[scikit-learn]: https://scikit-learn.org/stable/
[sklearn.base.BaseEstimator]: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
[sklearn.base.ClassifierMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html
[sklearn.base.RegressorMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html