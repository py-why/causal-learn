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
    G = cdnod(data, c_indx, alpha, indep_test, stable, uc_rule, uc_priority, mvcdnod,
          correction_name, background_knowledge, verbose, show_progress)

    # visualization using pydot
    # note that the last node is the c_indx
    cg.draw_pydot_graph()

Visualization using pydot is recommended. If specific label names are needed, please refer to this `usage example <https://github.com/cmu-phil/causal-learn/blob/e4e73f8b58510a3cd5a9125ba50c0ac62a425ef3/tests/TestGraphVisualization.py#L106>`_ (e.g., GraphUtils.to_pydot(cg.G, labels=["A", "B", "C"]).


Parameters
-------------------
**data**:  numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**c_indx**: time index or domain index that captures the unobserved changing factors.

**alpha**: desired significance level (float) in (0, 1). Default: 0.05.

**indep_test**: Independence test method function. Default: 'fisherz'.
       - ":ref:`fisherz <Fisher-z test>`": Fisher's Z conditional independence test.
       - ":ref:`chisq <Chi-Square test>`": Chi-squared conditional independence test.
       - ":ref:`gsq <G-Square test>`": G-squared conditional independence test.
       - ":ref:`kci <Kernel-based conditional independence (KCI) test and independence test>`": kernel-based conditional independence test. (As a kernel method, its complexity is cubic in the sample size, so it might be slow if the same size is not small.)
       - ":ref:`mv_fisherz <Missing-value Fisher-z test>`": Missing-value Fisher's Z conditional independence test.

**stable**: run stabilized skeleton discovery if True. Default: True.

**uc_rule**: how unshielded colliders are oriented. Default: 0.
       - 0: run uc_sepset.
       - 1: run maxP. Orient an unshielded triple X-Y-Z as a collider with an aditional CI test.
       - 2: run definiteMaxP. Orient only the definite colliders in the skeleton and keep track of all the definite non-colliders as well.

**uc_priority**: rule of resolving conflicts between unshielded colliders. Default: 2.
       - -1: whatever is default in uc_rule.
       - 0: overwrite.
       - 1: orient bi-directed.
       - 2: prioritize existing colliders.
       - 3: prioritize stronger colliders.
       - 4: prioritize stronger* colliders.

**mvpc**: use missing-value PC or not. Default (and suggested for CDNOD): False.

**correction_name**: Missing value correction if using missing-value PC. Default: 'MV_Crtn_Fisher_Z'

**background_knowledge**: class BackgroundKnowledge. Add prior edges according to assigned causal connections. Default: Nnoe.
For detailed usage, please kindly refer to its `usage example <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestBackgroundKnowledge.py>`_.

**verbose**: True iff verbose output should be printed. Default: False.

**show_progress**: True iff the algorithm progress should be show in console. Default: True.

Returns
-------------------
**cg** : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate  i --> j; cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j; cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

.. [1] Huang, B., Zhang, K., Zhang, J., Ramsey, J. D., Sanchez-Romero, R., Glymour, C., & Schölkopf, B. (2020). Causal Discovery from Heterogeneous/Nonstationary Data. J. Mach. Learn. Res., 21(89), 1-53.
