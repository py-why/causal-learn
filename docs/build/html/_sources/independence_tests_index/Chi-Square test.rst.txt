.. _Chi-Square test:

Chi-Square test
====================

Perform an independence test on discrete variables using Chi-Square test.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import chisq
    p = chisq(data, X, Y, conditioning_set)


Parameters
----------------
**data**: data matrices.

**X, Y and condition_set**: data matrices of size number_of_samples * dimensionality.

**G_sq**: True means using G-Square test;
       False means using Chi-Square test.

Returns
-------------
**p**: the p-value of the test.