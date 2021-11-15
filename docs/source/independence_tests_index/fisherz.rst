.. _Fisher-z test:

Fisher-z test
===================================

Perform an independence test using Fisher-z's test [1]_. This test is optimal for linear-Gaussian data.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import fisherz
    p = fisherz(data, X, Y, condition_set, correlation_matrix)

Parameters
------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**X, Y and condition_set**: column indices of data.

**correlation_matrix**: correlation matrix; None means without the parameter of correlation matrix.

Returns
-------------
**p**: the p-value of the test.

.. [1] Fisher, R. A. (1921). On the'probable error'of a coefficient of correlation deduced from a small sample. Metron, 1, 1-32.