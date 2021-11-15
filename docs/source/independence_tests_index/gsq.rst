.. _G-Square test:

G-Square test
================

Perform an independence test using G-Square test [1]_. This test is based on the log likelihood ratio test.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import gsq
    p = gsq(data, X, Y, conditioning_set)

Parameters
-------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**X, Y and condition_set**: column indices of data.

**G_sq**: True means using G-Square test; False means using Chi-Square test.

Returns
---------------
p: the p-value of the test

.. [1] Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian network structure learning algorithm. Machine learning, 65(1), 31-78.

