.. _Chi-Square test:

Chi-Square test
====================

Perform an independence test on discrete variables using Chi-Square test.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import cit
    p = cit.CIT(data, 'chisq')


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