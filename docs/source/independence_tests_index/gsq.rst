.. _G-Square test:

G-Square test
================

Perform an independence test using G-Square test [1]_. This test is based on the log likelihood ratio test.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import CIT
    gsq_obj = CIT(data, "gsq") # construct a CIT instance with data and method name
    pValue = gsq_obj(X, Y, S)

Please be kindly informed that we have refactored the independence tests from functions to classes since the release `v0.1.2.8 <https://github.com/cmu-phil/causal-learn/releases/tag/0.1.2.8>`_. Speed gain and a more flexible parameters specification are enabled.

For users, you may need to adjust your codes accordingly. Specifically, if you are

+ running a constraint-based algorithm from end to end: then you don't need to change anything. Old codes are still compatible. For example,
.. code-block:: python

    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import gsq
    cg = pc(data, 0.05, gsq)

+ explicitly calculating the p-value of a test: then you need to declare the :code:`gsq_obj` and then call it as above, instead of using :code:`gsq(data, X, Y, condition_set)` as before. Note that now :code:`causallearn.utils.cit.gsq` is a string :code:`"gsq"`, instead of a function.

Please see `CIT.py <https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/cit.py>`_
for more details on the implementation of the (conditional) independent tests.

Parameters
-------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**method**: string, "gsq".

**kwargs**: e.g., :code:`cache_path`. See :ref:`Advanced Usages <Advanced Usages>`.

Returns
---------------
p: the p-value of the test

.. [1] Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian network structure learning algorithm. Machine learning, 65(1), 31-78.

