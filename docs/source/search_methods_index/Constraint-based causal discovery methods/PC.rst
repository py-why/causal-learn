.. _pc:

PC
==

Algorithm Introduction
--------------------------------------

Perform Peter-Clark (PC [1]_) algorithm for causal discovery. We also allowed data sets with missing values,
for which testwise-deletion PC is included (choosing ‘MV-Fisher_Z” for the test name).

If you would like to use missing-value PC [2]_, please set 'mvpc' as True.


Usage
----------------------------
.. code-block:: python

    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(data, alpha, indep_test, stable, uc_rule, uc_priority, mvpc, correction_name, background_knowledge, verbose, show_progress)

    # visualization using pydot
    cg.draw_pydot_graph()

    # visualization using networkx
    # cg.to_nx_graph()
    # cg.draw_nx_graph(skel=False)

Visualization using pydot is recommended. If specific label names are needed, please refer to this `usage example <https://github.com/cmu-phil/causal-learn/blob/e4e73f8b58510a3cd5a9125ba50c0ac62a425ef3/tests/TestGraphVisualization.py#L106>`_ (e.g., GraphUtils.to_pydot(cg.G, labels=["A", "B", "C"]).

Parameters
-------------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

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

**mvpc**: use missing-value PC or not. Default: False.

**correction_name**. Missing value correction if using missing-value PC. Default: 'MV_Crtn_Fisher_Z'

**background_knowledge**: class BackgroundKnowledge. Add prior edges according to assigned causal connections. Default: None.
For detailed usage, please kindly refer to its `usage example <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestBackgroundKnowledge.py>`_.

**verbose**: True iff verbose output should be printed. Default: False.

**show_progress**: True iff the algorithm progress should be show in console. Default: True.


Returns
-------------------
**cg** : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate  i --> j; cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicate i --- j; cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

.. [1] Spirtes, P., Glymour, C. N., Scheines, R., & Heckerman, D. (2000). Causation, prediction, and search. MIT press.
.. [2] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellström, H., & Zhang, K. (2019, April). Causal discovery in the presence of missing data. In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 1762-1770). PMLR.