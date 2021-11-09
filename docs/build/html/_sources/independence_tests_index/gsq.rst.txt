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
**data**: data matrices.

**X, Y and condition_set**: data matrices of size number_of_samples * dimensionality.

**G_sq**: True means using G-Square test; False means using Chi-Square test.

Returns
---------------
p: the p-value of the test

.. [1] Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian network structure learning algorithm. Machine learning, 65(1), 31-78.

