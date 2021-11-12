.. _pnl:

Post-nonlinear causal models
=============================

Algorithm Introduction
--------------------------------------

Causal discovery based on the post-nonlinear (PNL [1]_) causal models.
If you would like to apply the method to more than two variables,
we suggest you first apply the PC algorithm and then use pair-wise
analysis in this implementation to find the causal directions that
cannot be determined by PC.

Usage
-------------

.. code-block:: python

    from causallearn.search.FCMBased.PNL.PNL import PNL
    pnl = PNL()
    p_value_foward, p_value_backward = pnl.cause_or_effect(data_x, data_y)

Parameters
--------------------------------------

**data_x**: input data (n, 1), n is the sample size.

**data_y**: output data (n, 1), n is the sample size.

Returns
--------------------------------------

**pval_forward**: p value in the x->y direction.

**pval_backward**: p value in the y->x direction.

.. [1] Zhang, K., & Hyvärinen, A. (2009, June). On the Identifiability of the Post-Nonlinear Causal Model. In 25th Conference on Uncertainty in Artificial Intelligence (UAI 2009) (pp. 647-655). AUAI Press.
