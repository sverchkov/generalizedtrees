# Package Architecture 0.0.2

**This is the plan for the architecture of version 0.0.2. It is not reflective of the current architecture.**

The main purpose of this package is to serve as a library that provides [ScikitLearn]-compatible tree classifiers and
regressors.
Specifically we provide the following implementations:

* Standard decision tree classifier
* Logistic regression-based model tree classifier
* Simple mimic tree classifier
* More to come

To be a [ScikitLearn]-compatible classifier (regressor), a model must be implemented as a class that inherits from the
[sklearn.base.BaseEstimator] base class and the [sklearn.base.ClassifierMixin] ([sklearn.base.RegressorMixin]) mixin.

We implement high-level decision/regression tree -building logic in the abstract class [abstract.AbstractTreeEstimator].
All concrete tree estimators therefore have the following declaration of inheritance:

```python
    from sklearn.base import BaseEstimator, ClassifierMixin
    from generalizedtrees.abstract import AbstractTreeEstimator
    
    class OurTreeEstimator(BaseEstimator, ClassifierMixin, AbstractTreeEstimator):
    
        # Implementation
```

## The Abstract Tree Estimator
Implements: (TODO)
Abstract class contract requests implementation of: (TODO)

## Other abstract classes

### The Constraint object hierarchy

TODO

---
Licensed under the Apache License, Version 2.0, Copyright 2019 Yuriy Sverchkov

[ScikitLearn]: https://scikit-learn.org/stable/
[sklearn.base.BaseEstimator]: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
[sklearn.base.ClassifierMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html
[sklearn.base.RegressorMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html