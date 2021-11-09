.. _Missing-value Fisher-z test:

Missing-value Fisher-z test
====================================

Perform a testwise-deletion Fisher-z independence test to data sets with missing values.
With testwise-deletion, the test makes use of all data points that do not have missing values for the variables involved in the test.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import mv_fisherz
    p = mv_fisherz(mvdata, X, Y, condition_set)


Parameters
---------------
**mvdata**: data with missing values.

**X, Y and condition_set**: data matrices of size number_of_samples * dimensionality.

Returns
----------------
**p**: the p-value of the test.
