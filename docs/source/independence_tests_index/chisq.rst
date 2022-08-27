.. _Chi-Square test:

Chi-Square test
====================

Perform an independence test on discrete variables using Chi-Square test.

(We have updated the independence test class and the usage example hasn't been updated yet. For new class, please refer to `TestCIT.py <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestCIT.py>`_ or `TestCIT_KCI.py <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestCIT_KCI.py>`_.)

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import chisq
    p = chisq(data, X, Y, conditioning_set)


Parameters
----------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**X, Y and condition_set**: column indices of data.

**G_sq**: True means using G-Square test;
       False means using Chi-Square test.

Returns
-------------
**p**: the p-value of the test.