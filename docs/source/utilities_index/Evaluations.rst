.. _evaluation:

Evaluations
==============================================



Usage
----------------------------
.. code-block:: python

    from causallearn.graph.ArrowConfusion import ArrowConfusion
    from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
    from causallearn.graph.SHD import SHD

    # For arrows
    arrow = ArrowConfusion(truth_cpdag, est)

    arrowsTp = arrow.get_arrows_tp()
    arrowsFp = arrow.get_arrows_fp()
    arrowsFn = arrow.get_arrows_fn()
    arrowsTn = arrow.get_arrows_tn()

    arrowPrec = arrow.get_arrows_precision()
    arrowRec = arrow.get_arrows_recall()

    # For adjacency matrices
    adj = AdjacencyConfusion(truth_cpdag, est)

    adjTp = adj.get_adj_tp()
    adjFp = adj.get_adj_fp()
    adjFn = adj.get_adj_fn()
    adjTn = adj.get_adj_tn()

    adjPrec = adj.get_adj_precision()
    adjRec = adj.get_adj_recall()

    # Structural Hamming Distance
    shd = SHD(truth_cpdag, est).get_shd()

Parameters
-------------------
**X**: Data with T*D dimensions.

**truth_cpdag**: Graph class.

**est**: Graph class.

Returns
-------------------

**arrowsTp/Fp/Fn/Tn**: True positive/false positive/false negative/true negative arrows.

**arrowPrec**: Precision for arrows.

**arrowRec**: Recall for arrows.

**adjTp/Fp/Fn/Tn**: True positive/false positive/false negative/true negative edges.

**adjPrec**: Precision for the adjacency matrix.

**adjRec**: Recall for the adjacency matrix.

**shd**: Structural Hamming Distance.

