.. _rlcd:

Rank-based Latent Causal Discovery (RLCD)
=============================================================

Algorithm Introduction
-----------------------------------------------------------

RLCD [1]_ learns causal structures with causally-related hidden variables from rank constraints in partially observed linear causal models.


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

Inspecting latent variables
-----------------------------------------------------------
The returned ``CausalGraph`` includes both observed variables and detected latent variables. Observed variables appear first, followed by latent variables named ``L1``, ``L2``, ...

.. code-block:: python

    from causallearn.graph.NodeType import NodeType

    latent_nodes = [
        node for node in cg.G.get_nodes()
        if node.get_node_type() == NodeType.LATENT
    ]

    print([node.get_name() for node in latent_nodes])
    print(cg.all_vars)

RLCD also attaches the following outputs to the returned graph:

.. code-block:: python

    cg.stage1_cg   # stage-1 graph over observed variables
    cg.adjacency   # adjacency matrix including observed and latent variables
    cg.all_vars    # observed variables followed by detected latent variables

For example, the following data has five observed variables generated from one shared latent variable. RLCD can add the detected latent variable to the returned graph.

.. code-block:: python

    import numpy as np
    from causallearn.graph.NodeType import NodeType
    from causallearn.search.HiddenCausal.RLCD import Chi2RankTest, RLCD

    rng = np.random.default_rng(1)
    sample_size = 3000
    latent = rng.normal(size=sample_size)
    data = np.column_stack([
        1.0 * latent + 0.05 * rng.normal(size=sample_size),
        1.2 * latent + 0.05 * rng.normal(size=sample_size),
        1.4 * latent + 0.05 * rng.normal(size=sample_size),
        1.6 * latent + 0.05 * rng.normal(size=sample_size),
        1.8 * latent + 0.05 * rng.normal(size=sample_size),
    ])
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    cg = RLCD(
        data,
        ranktest_method=Chi2RankTest(data),
        stage1_method="all",
        maxk=2,
    )

    latent_nodes = [
        node for node in cg.G.get_nodes()
        if node.get_node_type() == NodeType.LATENT
    ]

    print(cg.all_vars)
    print([node.get_name() for node in latent_nodes])

This example prints ``['X1', 'X2', 'X3', 'X4', 'X5', 'L1']`` for ``cg.all_vars`` and ``['L1']`` for the detected latent variables.

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
**cg**: CausalGraph. Learned graph over observed and latent variables, where ``cg.G.graph[j,i]=1`` and ``cg.G.graph[i,j]=-1`` indicate ``i --> j``; ``cg.G.graph[i,j] = cg.G.graph[j,i] = -1`` indicate ``i --- j``; ``cg.G.graph[i,j] = cg.G.graph[j,i] = 1`` indicates ``i <-> j``. The returned object also stores ``cg.stage1_cg``, ``cg.adjacency``, and ``cg.all_vars`` for inspecting the stage-1 graph, the full adjacency matrix, and the variable names including latent variables.

.. [1] Dong, X., Huang, B., Ng, I., Song, X., Zheng, Y., Jin, S., Legaspi, R., Spirtes, P., & Zhang, K. (2024). A versatile causal discovery framework to allow causally-related hidden variables. In International Conference on Learning Representations, vol. 2024, pp. 43084-43118.
