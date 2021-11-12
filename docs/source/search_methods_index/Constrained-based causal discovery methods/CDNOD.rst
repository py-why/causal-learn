.. _cdnod:

CD-NOD
=======

Algorithm Introduction
--------------------------------------

Perform Peter-Clark algorithm for causal discovery on the augmented data set that captures the unobserved changing factors (CD-NOD, [1]_).


Usage
----------------------------
.. code-block:: python

    from causallearn.search.ConstraintBased.CDNOD import cdnod
    G = cdnod(data, c_indx, alpha, indep_test, stable, uc_rule, uc_priority, mvpc, correction_name)
    G.to_nx_graph()
    G.draw_nx_graph(skel=False)

Parameters
-------------------
**data**:  numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**c_indx**: time index or domain index that captures the unobserved changing factors.

**alpha**: desired significance level (float) in (0, 1).

**indep_test**: Independence test method function.
       - ":ref:`fisherz <Fisher-z test>`": Fisher's Z conditional independence test.
       - ":ref:`chisq <Chi-Square test>`": Chi-squared conditional independence test.
       - ":ref:`gsq <G-Square test>`": G-squared conditional independence test.
       - ":ref:`kci <Kernel-based conditional independence (KCI) test and independence test>`": kernel-based conditional independence test. (As a kernel method, its complexity is cubic in the sample size, so it might be slow if the same size is not small.)
       - ":ref:`mv_fisherz <Missing-value Fisher-z test>`": Missing-value Fisher's Z conditional independence test.

**stable**: run stabilized skeleton discovery if True (default = True).

**uc_rule**: how unshielded colliders are oriented.
       - 0: run uc_sepset.
       - 1: run maxP. Orient an unshielded triple X-Y-Z as a collider with an aditional CI test.
       - 2: run definiteMaxP. Orient only the definite colliders in the skeleton and keep track of all the definite non-colliders as well.

**uc_priority**: rule of resolving conflicts between unshielded colliders.
       - -1: whatever is default in uc_rule.
       - 0: overwrite.
       - 1: orient bi-directed.
       - 2: prioritize existing colliders.
       - 3: prioritize stronger colliders.
       - 4: prioritize stronger* colliders.

**mvpc**: use missing-value PC or not. Default (and suggested for CDNOD): False.

**correction_name**. Missing value correction if using missing-value PC. Default: 'MV_Crtn_Fisher_Z'

Returns
-------------------
**cg** : a CausalGraph object. Nodes in the graph correspond to the column indices in the data.

.. [1] Huang, B., Zhang, K., Zhang, J., Ramsey, J. D., Sanchez-Romero, R., Glymour, C., & Schölkopf, B. (2020). Causal Discovery from Heterogeneous/Nonstationary Data. J. Mach. Learn. Res., 21(89), 1-53.
