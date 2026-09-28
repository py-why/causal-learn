.. _xges:

XGES: Extremely Greedy Equivalence Search
==========================================

Algorithm Introduction
--------------------------------------

Extremely Greedy Equivalence Search (XGES) [1]_ is a score-based causal discovery
algorithm that learns a CPDAG (a Markov equivalence class) from observational data.
Like GES, it optimizes a decomposable score (e.g. BIC), but instead of GES's separate
forward (insert-only) and backward (delete-only) phases, XGES maintains **all** candidate
edits together and, at every step, applies the single globally-best-scoring edit across
three operators:

- **Insert(x, y, T)**: add ``x -> y`` and orient ``t - y`` as ``t -> y`` for ``t`` in ``T``.
- **Delete(x, y, H)**: remove ``x - y`` / ``x -> y`` and orient the members of ``H``.
- **Reverse(x, y, T)**: turn an existing ``y -> x`` into ``x -> y`` (and orient ``t -> y``
  for ``t`` in ``T``). This operator is **not** present in GES.

The three operators are kept in score-delta-ordered candidate sets; after each applied
edit the graph is re-canonicalized to a CPDAG and the affected candidates are re-evaluated.
XGES prioritizes deletions, then reversals, then insertions, which favors removing edges
early in the search and reduces the chance of ending in a poor local optimum.

Two variants are selectable:

- **XGES-0** (``extended_search=False``): the pure three-operator greedy loop.
- **XGES** (``extended_search=True``, default): XGES-0 plus a local-optima-escape step
  that forces each valid deletion, re-runs the greedy search (forbidding re-insertion of
  the deleted edge), and keeps the result only when it strictly improves the score.

This implementation is **clean-room from the paper (arXiv:2502.19551) only**; it reuses
causal-learn's BIC local score and PDAG/DAG converters and adds no new dependency beyond
numpy/scipy.


Usage
----------------------------
.. code-block:: python

    from causallearn.search.ScoreBased.XGES import xges

    # Basic usage with default BIC score (full XGES with local-optima escape)
    Record = xges(X)

    # XGES-0: the pure greedy loop, without the extended local-optima search
    Record = xges(X, extended_search=False)

    # With custom BIC penalty (lambda_value) and node names
    Record = xges(X, score_func='local_score_BIC',
                  parameters={'lambda_value': 1.0},
                  node_names=['X1', 'X2', 'X3'])

    # Using the paper's BIC penalty parameter alpha (mapped to lambda_value = alpha / 2).
    # The paper's experiments use alpha = 2.0, i.e. lambda_value = 1.0.
    Record = xges(X, score_func='local_score_BIC', alpha=2.0)

    # Visualization using pydot
    from causallearn.utils.GraphUtils import GraphUtils
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import io

    pyd = GraphUtils.to_pydot(Record['G'])
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()


Parameters
-------------------
**X**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features. May be ``None`` if ``cov`` and ``n`` are provided (BIC only).

**score_func**: The score function to use. Default: ``'local_score_BIC'``. The same set of score
functions as GES is supported (``'local_score_BIC'``, ``'local_score_BIC_from_cov'``, ``'local_score_BDeu'``,
``'local_score_CV_general'``, ``'local_score_marginal_general'``, ``'local_score_CV_multi'``,
``'local_score_marginal_multi'``).

**maxP**: Allowed maximum number of parents when searching the graph. Default: None (uses the number of features).

**parameters**: Additional parameters for the score function. Default: None.
              - parameters['lambda_value']: Penalty hyperparameter for BIC score. Default: 0.5.

**node_names**: list of str or None. Custom names for graph nodes. Default: None (uses ``X1, X2, ...``).

**cov**, **n**: numpy.ndarray and int. Covariance matrix and sample size, as an alternative to ``X`` for BIC scores.

**lambda_value**: float. BIC penalty coefficient (``lambda_value * (|Pa| + 1) * log(n)`` per node). Default: 0.5.

**alpha**: float or None. The paper's BIC penalty (``S = log-likelihood - (alpha / 2) * log(n) * |theta|``).
When provided (and ``lambda_value`` is not), it is mapped to ``lambda_value = alpha / 2``, so the paper
default ``alpha = 2.0`` corresponds to ``lambda_value = 1.0`` and the textbook BIC ``alpha = 1.0`` to
``lambda_value = 0.5``. Default: None.

**extended_search**: bool. If True (default), run full XGES (with the local-optima-escape step);
if False, run only the XGES-0 greedy core.


Returns
-------------------
- **Record['G']**: GeneralGraph. The learned causal graph (CPDAG), where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicate i --> j; Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.

- **Record['score']**: float. The score of the learned graph.

.. [1] Nazaret, A., & Blei, D. (2024). Extremely Greedy Equivalence Search. In Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence (UAI). arXiv:2502.19551.
