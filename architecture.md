# Package Architecture 0.0.2

The main purpose of this package is to serve as a library that provides [Scikit Learn]-compatible tree classifiers and
regressors.
Specifically we provide the following implementations:

* Standard decision tree classifier
* Logistic regression-based model tree classifier
* Simple mimic tree classifier
* More to come

To be a [Scikit Learn]-compatible classifier (regressor), a model must be implemented as a class that inherits from the
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

---
Licensed under the Apache License, Version 2.0, Copyright 2019 Yuriy Sverchkov

[Scikit Learn]: https://scikit-learn.org/stable/
[sklearn.base.BaseEstimator]: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
[sklearn.base.ClassifierMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html
[sklearn.base.RegressorMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html