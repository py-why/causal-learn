.. _rlcd:

Rank-based Latent Causal Discovery (RLCD)
=============================================================

Algorithm Introduction
-----------------------------------------------------------

RLCD [1]_ learns causal structures with causally-related hidden variables from rank constraints in partially observed linear causal models.

This implementation includes the structure learning part of RLCD from ``scm-identify``. It provides the main RLCD search routine and the rank-test helper used for sample data.

Usage
-----------------------------------------------------------
.. code-block:: python

    from causallearn.search.HiddenCausal.RLCD import RLCD

    # default parameters
    cg = RLCD(data)

    # or customized parameters
    cg = RLCD(data, ranktest_method, stage1_method, alpha_dict, maxk, node_names)

    # visualization using pydot
    cg.draw_pydot_graph()

    # or save the graph
    from causallearn.utils.GraphUtils import GraphUtils

    pyd = GraphUtils.to_pydot(cg.G)
    pyd.write_png('rlcd_result.png')

Visualization using pydot is recommended. If specific label names are needed, please refer to this `usage example <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestGraphVisualization.py>`_ (e.g., 'cg.draw_pydot_graph(labels=["A", "B", "C"])' or 'GraphUtils.to_pydot(cg.G, labels=["A", "B", "C"])').

Parameters
-----------------------------------------------------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**ranktest_method**: rank test object, optional. The rank test object should provide a ``test(pcols, qcols, r, alpha)`` method. If not provided, ``Chi2RankTest(data)`` is used.

**stage1_method**: str. Stage-1 method used to partition observed variables. Default: 'ges'.

**alpha_dict**: dict, optional. Significance levels for rank tests by rank. Default: ``{0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01}``.

**maxk**: int. Maximum rank-search cardinality. Default: 3.

**node_names**: list, optional. Names of observed variables in the returned graph. If not provided, variables are named ``X1``, ``X2``, ... Latent variables are named ``L1``, ``L2``, ...

Returns
-----------------------------------------------------------
**cg**: CausalGraph. Learned graph over observed and latent variables, where ``cg.G.graph[j,i]=1`` and ``cg.G.graph[i,j]=-1`` indicate ``i --> j``; ``cg.G.graph[i,j] = cg.G.graph[j,i] = -1`` indicate ``i --- j``; ``cg.G.graph[i,j] = cg.G.graph[j,i] = 1`` indicates ``i <-> j``.

.. [1] Dong, X., Huang, B., Ng, I., Song, X., Zheng, Y., Jin, S., Legaspi, R., Spirtes, P., & Zhang, K. (2024). A versatile causal discovery framework to allow causally-related hidden variables. In International Conference on Learning Representations, vol. 2024, pp. 43084-43118.
