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

    # default parameters
    cg = pc(data)

    # or customized parameters
    cg = pc(data, alpha, indep_test, stable, uc_rule, uc_priority, mvpc, correction_name, background_knowledge, verbose, show_progress)

    # visualization using pydot
    cg.draw_pydot_graph()

    # or save the graph
    from causallearn.utils.GraphUtils import GraphUtils

    pyd = GraphUtils.to_pydot(cg.G)
    pyd.write_png('simple_test.png')

    # visualization using networkx
    # cg.to_nx_graph()
    # cg.draw_nx_graph(skel=False)

Visualization using pydot is recommended. If specific label names are needed, please refer to this `usage example <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestGraphVisualization.py>`_ (e.g., 'cg.draw_pydot_graph(labels=["A", "B", "C"])' or 'GraphUtils.to_pydot(cg.G, labels=["A", "B", "C"])').

.. _Advanced Usages:

Advanced Usages
----------------------------
+ If you would like to specify parameters for the (conditional) independence test (if available), you may directly pass the parameters to the :code:`pc` call. E.g.,

  .. code-block:: python

    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import kci
    cg = pc(data, 0.05, kci, kernelZ='Polynomial', approx=False, est_width='median', ...)

+ If your graph is big and/or your independence test is slow (e.g., KCI), you may want to cache the p-value results to a local checkpoint. Then by reading values from this local checkpoint, no more repeated calculation will be wasted to resume from checkpoint / just finetune some PC parameters. This can be achieved by specifying :code:`cache_path`. E.g.,

  .. code-block:: python

        citest_cache_file = "/my/path/to/citest_cache_dataname_kci.json"    # .json file
        cg1 = pc(data, 0.05, kci, cache_path=citest_cache_file)             # after the long run

        # just finetune uc_rule. p-values are reused, and thus cg2 is done in almost no time.
        cg2 = pc(data, 0.05, kci, cache_path=citest_cache_file, uc_rule=1)
  ..

  If :code:`cache_path` does not exist in your local file system, a new one will be created. Otherwise, the cache will be first loaded from the json file to the CIT class and used during the runtime. Note that 1) data hash and parameters hash will first be checked at loading to ensure consistency, and 2) during runtime, the cache will be saved to the local file every 30 seconds.

+ The above advanced usages also apply to other constraint-based methods, e.g., FCI and CDNOD.


Parameters
-------------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**alpha**: desired significance level (float) in (0, 1). Default: 0.05.

**indep_test**: string, name of the independence test method. Default: 'fisherz'.
       - ":ref:`fisherz <Fisher-z test>`": Fisher's Z conditional independence test.
       - ":ref:`chisq <Chi-Square test>`": Chi-squared conditional independence test.
       - ":ref:`gsq <G-Square test>`": G-squared conditional independence test.
       - ":ref:`kci <Kernel-based conditional independence (KCI) test and independence test>`": kernel-based conditional independence test. (As a kernel method, its complexity is cubic in the sample size, so it might be slow if the same size is not small.)
       - ":ref:`mv_fisherz <Missing-value Fisher-z test>`": Missing-value Fisher's Z conditional independence test.

**stable**: run stabilized skeleton discovery [4]_ if True. Default: True.

**uc_rule**: how unshielded colliders are oriented. Default: 0.
       - 0: run uc_sepset.
       - 1: run maxP [3]_. Orient an unshielded triple X-Y-Z as a collider with an additional CI test.
       - 2: run definiteMaxP [3]_. Orient only the definite colliders in the skeleton and keep track of all the definite non-colliders as well.

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
.. [3] Ramsey, J. (2016). Improving accuracy and scalability of the pc algorithm by maximizing p-value. arXiv preprint arXiv:1610.00378.
.. [4] Colombo, D., & Maathuis, M. H. (2014). Order-independent constraint-based causal structure learning. J. Mach. Learn. Res., 15(1), 3741-3782.