.. _GRaSP:

GRaSP
==============================================

Algorithm Introduction
--------------------------------------

Greedy relaxation of the sparsest permutation (GRaSP) algorithm [1]_.


Usage
----------------------------
.. code-block:: python

    from causallearn.search.PermutationBased.GRaSP import grasp

    # default parameters
    G = grasp(X)

    # or customized parameters
    G = grasp(X, score_func, depth, maxP, parameters)

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

Visualization using pydot is recommended (`usage example <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestGRaSP.py>`_). If specific label names are needed, please refer to this `usage example <https://github.com/cmu-phil/causal-learn/blob/e4e73f8b58510a3cd5a9125ba50c0ac62a425ef3/tests/TestGraphVisualization.py#L106>`_ (e.g., GraphUtils.to_pydot(G, labels=["A", "B", "C"]).

Parameters
-------------------
**X**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**score_func**: The score function you would like to use, including (see :ref:`score_functions`.). Default: 'local_score_BIC'.
              - ":ref:`local_score_BIC <BIC score>`": BIC score [3]_.
              - ":ref:`local_score_BDeu <BDeu score>`": BDeu score [4]_.
              - ":ref:`local_score_CV_general <Generalized score with cross validation>`": Generalized score with cross validation for data with single-dimensional variables [2]_.
              - ":ref:`local_score_marginal_general <Generalized score with marginal likelihood>`": Generalized score with marginal likelihood for data with single-dimensional variables [2]_.
              - ":ref:`local_score_CV_multi <Generalized score with cross validation>`": Generalized score with cross validation for data with multi-dimensional variables [2]_.
              - ":ref:`local_score_marginal_multi <Generalized score with marginal likelihood>`": Generalized score with marginal likelihood for data with multi-dimensional variables [2]_.

**maxP**: Allowed maximum number of parents when searching the graph. Default: None.

**parameters**: Needed when using CV likelihood. Default: None.
              - parameters['kfold']: k-fold cross validation.
              - parameters['lambda']: regularization parameter.
              - parameters['dlabel']: for variables with multi-dimensions, indicate which dimensions belong to the i-th variable.



Returns
-------------------
- **G**: learned general graph, where G.graph[j,i]=1 and G.graph[i,j]=-1 indicate i --> j; G.graph[i,j] = G.graph[j,i] = -1 indicates i --- j.


.. [1] Lam, W. Y., Andrews, B., & Ramsey, J. (2022, February). Greedy Relaxations of the Sparsest Permutation Algorithm. In The 38th Conference on Uncertainty in Artificial Intelligence.
.. [2] Huang, B., Zhang, K., Lin, Y., Schölkopf, B., & Glymour, C. (2018, July). Generalized score functions for causal discovery. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 1551-1560).
.. [3] Schwarz, G. (1978). Estimating the dimension of a model. The annals of statistics, 461-464.
.. [4] Buntine, W. (1991). Theory refinement on Bayesian networks. In Uncertainty proceedings 1991 (pp. 52-60). Morgan Kaufmann.
