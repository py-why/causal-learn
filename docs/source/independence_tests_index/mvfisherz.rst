.. _Missing-value Fisher-z test:

Missing-value Fisher-z test
====================================

Perform a testwise-deletion Fisher-z independence test to data sets with missing values.
With testwise-deletion, the test makes use of all data points that do not have missing values for the variables involved in the test.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import CIT
    mv_fisherz_obj = CIT(data_with_missingness, "mv_fisherz") # construct a CIT instance with data and method name
    pValue = mv_fisherz_obj(X, Y, S)

Please be kindly informed that we have refactored the independence tests from functions to classes since the release `v0.1.2.8 <https://github.com/cmu-phil/causal-learn/releases/tag/0.1.2.8>`_. Speed gain and a more flexible parameters specification are enabled.

For users, you may need to adjust your codes accordingly. Specifically, if you are

+ running a constraint-based algorithm from end to end: then you don't need to change anything. Old codes are still compatible. For example,
.. code-block:: python

    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import mv_fisherz
    cg = pc(data_with_missingness, 0.05, mv_fisherz)

+ explicitly calculating the p-value of a test: then you need to declare the :code:`mv_fisherz_obj` and then call it as above, instead of using :code:`mv_fisherz(data, X, Y, condition_set)` as before. Note that now :code:`causallearn.utils.cit.mv_fisherz` is a string :code:`"mv_fisherz"`, instead of a function.

Please see `CIT.py <https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/cit.py>`_
for more details on the implementation of the (conditional) independent tests.


Parameters
------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**method**: string, "mv_fisherz".

**kwargs**: e.g., :code:`cache_path`. See :ref:`Advanced Usages <Advanced Usages>`.

Returns
----------------
**p**: the p-value of the test.
