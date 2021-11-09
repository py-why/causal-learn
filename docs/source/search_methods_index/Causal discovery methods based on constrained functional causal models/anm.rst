.. _anm:

Additive noise models
=============================

Algorithm Introduction
--------------------------------------

Causal discovery based on the additive noise models (ANM [1]_).
If you would like to apply the method to more than two variables,
we suggest you first apply the PC algorithm and then use pair-wise
analysis in this implementation to find the causal directions that
cannot be determined by PC.

Usage
-------------

.. code-block:: python

    from causallearn.search.FCMBased.ANM.ANM import ANM
    anm = ANM()
    p_value_foward, p_value_backward = anm.cause_or_effect(data_x, data_y)


Parameters
--------------------------------------

**data_x**: input data (n, 1).

**data_y**: output data (n, 1).

Returns
--------------------------------------

**pval_forward**: p value in the x->y direction.

**pval_backward**: p value in the y->x direction.


.. [1] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2008, December). Nonlinear causal discovery with additive noise models. In NIPS (Vol. 21, pp. 689-696).