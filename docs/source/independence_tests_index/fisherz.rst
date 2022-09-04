.. _Fisher-z test:

Fisher-z test
===================================

Perform an independence test using Fisher-z's test [1]_. This test is optimal for linear-Gaussian data.


Usage
--------
.. code-block:: python

    from causallearn.utils.cit import CIT
    fisherz_obj = CIT(data, "fisherz") # construct a CIT instance with data and method name
    pValue = fisherz_obj(X, Y, S)

Please be kindly informed that we have refactored the independence tests from functions to classes since the release `v0.1.2.8 <https://github.com/cmu-phil/causal-learn/releases/tag/0.1.2.8>`_. Speed gain and a more flexible parameters specification are enabled.

For users, you may need to adjust your codes accordingly. Specifically,

+ If you are running a constraint-based algorithm from end to end: then you don't need to change anything. Old codes are still compatible. For example,
.. code-block:: python

    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    cg = pc(data, 0.05, fisherz)

+ If you are explicitly calculating the p-value of a test: then you need to declare the :code:`fisherz_obj` and then call it as above, instead of using :code:`fisherz(data, X, Y, condition_set)` as before. Note that now :code:`causallearn.utils.cit.fisherz` is a string :code:`"fisherz"`, instead of a function.


Please see `CIT.py <https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/cit.py>`_
for more details on the implementation of the (conditional) independent tests.

Parameters
------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**method**: string, "fisherz".

**kwargs**: e.g., :code:`cache_path`. See :ref:`Advanced Usages <Advanced Usages>`.

Returns
-------------
**p**: the p-value of the test.

.. [1] Fisher, R. A. (1921). On the'probable error'of a coefficient of correlation deduced from a small sample. Metron, 1, 1-32.