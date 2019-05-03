# Package Architecture 0.0.2

The main purpose of this package is to serve as a library that provides [Scikit Learn]-compatible tree classifiers and
regressors.
Specifically we provide the following implementations:

* Standard decision tree classifier
* Logistic regression-based model tree classifier
* Simple mimic tree classifier
* More to come

To be a [scikit-learn]-compatible classifier (regressor), a model must be implemented as a class that inherits from the
[sklearn.base.BaseEstimator] base class and the [sklearn.base.ClassifierMixin] ([sklearn.base.RegressorMixin]) mixin.

We implement high-level decision/regression tree -building logic in the abstract class [abstract.AbstractTreeEstimator].
All concrete tree estimators therefore have the following declaration of inheritance:

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from generalizedtrees.core import AbstractTreeEstimator
    
class OurTreeEstimator(BaseEstimator, ClassifierMixin, AbstractTreeEstimator):
    
    # Must implement:
    def leaf_predictor(self, constraints):
        pass
        
    def best_split(self, constraints):
        pass
    
    def fit(self, data, targets):  # or just data
        pass
```

## The Abstract Tree Estimator
Implements:

 * `predict(data)`

Abstract class contract requests implementation of:

 * `leaf_predictor(constraints)`
 * `best_split(constraints)`

## Other abstract classes

### The sequential access data structure
Used by `AbstractTreeEstimator`. TODO

### The Constraint object hierarchy

TODO

## Scikit-learn rules

To be able to pass the scikit-learn checks an estimator must:

 * All arguments of `__init__` must have a default value
 * All arguments of `__init__` must correspond to class attributes.
 * `__init__` cannot set attributes that do not correspond to its arguments.
 This forces us to declare fit parameters (named with a trailing underscore) outside `__init__`.
 To make thing nicer I do declare the types of the fit parameters in `__init__`, but we still get an inspection note
 for declaring parameters outside`__init__`.
 * The second parameter of `fit` must be `y` or `Y`.
 * See the following subsection for a description of what is considered to be part of the scikit-learn interface and
 suggestions for how to handle

### The scikit-learn interface

| Method | Defined in | Do we override
| --- | --- | ---
| `predict(data)` | `AbstractTreeEstimator` | Should not need to (we would adjust `AbstractTreeEstimator` if needed).
| `fit(data,[y]) | Concrete implementation |
| `score(data, y)` | `BaseEstimator` | We may want to (the default computes prediction error).
| `get_params` and `set_params` | `BaseEstimator` | Don't override.

### Other scikit-learn checks

In our unit tests we use `sklearn.utils.estimator_checks.check_estimator`.
Here we document some of the less-obvious checks it performs:

 * Passing an object to the `y` argument of `fit` should raise an exception.
 Use `sklearn.utils.multiclass.check_classification_targets` to ensure it.
 * `fit` returns `self`
 * Predict raises an error for infinite and NaN inputs. This is definitely not needed for decision trees, so we add a
 parameter to `predict` that tells us whether to really perform this check.
 * Classifiers must have a `classes_` element
 * Trying to predict without fitting must raise an error containing the word "fit"

---
Licensed under the Apache License, Version 2.0, Copyright 2019 Yuriy Sverchkov

[scikit-learn]: https://scikit-learn.org/stable/
[sklearn.base.BaseEstimator]: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
[sklearn.base.ClassifierMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html
[sklearn.base.RegressorMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html