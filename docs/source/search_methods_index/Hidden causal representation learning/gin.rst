.. _gin:

Generalized Independence Noise (GIN) condition-based method
=============================================================

Algorithm Introduction
-----------------------------------------------------------

Learning the structure of Linear, Non-Gaussian LAtent variable Model (LiNLAM) based the GIN [1]_ condition.

Usage
-----------------------------------------------------------
.. code-block:: python

    from causallearn.search.FCMBased.GIN.GIN import GIN
    G, K = GIN(data)

    # Visualization using pydot
    from causallearn.utils.GraphUtils import GraphUtils
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import io

    pyd = GraphUtils.to_pydot(G)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

Visualization using pydot is recommended (`usage example <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestGIN.py>`_). If specific label names are needed, please refer to this `usage example <https://github.com/cmu-phil/causal-learn/blob/e4e73f8b58510a3cd5a9125ba50c0ac62a425ef3/tests/TestGraphVisualization.py#L106>`_ (e.g., GraphUtils.to_pydot(G, labels=["A", "B", "C"]).

Parameters
-----------------------------------------------------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

Returns
-----------------------------------------------------------
**G**: GeneralGraph. Causal graph.

**K**: list. Causal Order.

.. [1] Xie, F., Cai, R., Huang, B., Glymour, C., Hao, Z., & Zhang, K. (2020, January). Generalized Independent Noise Condition for Estimating Latent Variable Causal Graphs. In NeurIPS.