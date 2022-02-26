Linear granger causality
==========================

Algorithm Introduction
--------------------------------------

Implementation of granger causality [1]_, including 1) regression+hypothesis test and 2) lasso regression.

Usage
----------------------------

.. code-block:: python

    from causallearn.search.Granger.Granger import Granger
    G = Granger()
    p_value_matrix = G.granger_test_2d(data)
    coeff = G.granger_lasso(data)

Parameters
-------------------

**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features. Note that for granger_test_2d(), the shape of input data is (n_samples, 2).

Returns
-------------------

**p_value_matrix**: p values for x1->x2 and x2->x1 (for 'granger_test_2d', which is the granger causality test for two-dimensional time series).

**coeff**: coefficient matrix (for 'granger_lasso', which is the granger causality test for multi-dimensional time series).

.. [1] Granger, C. W. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica: journal of the Econometric Society, 424-438.
