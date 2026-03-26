.. _dges:

DGES: GES in the Presence of Deterministic Relations
======================================================

Algorithm Introduction
--------------------------------------

Deterministic Greedy Equivalence Search (DGES) [1]_ extends GES to handle data containing
deterministic (functional) relationships among variables. Standard GES can fail when some
variables are deterministic functions of others, because the standard BIC score becomes
degenerate (the residual variance is zero). DGES addresses this by detecting
**Minimal Deterministic Clusters (MinDCs)** from the data and incorporating them into
the search procedure.

DGES implements Algorithm 1 from the paper, which consists of three phases:

- **Phase 1 — Detect MinDCs**: Identify sets of variables where one variable is a deterministic function of the others, using eigenvalue analysis of the covariance matrix.
- **Phase 2 — DC-aware GES**: Run a modified GES that forces edges between variables within the same MinDC during the forward phase and protects these edges from deletion during the backward phase. This uses a deterministic-aware BIC score that handles zero residual variance gracefully.
- **Phase 3 — Exact Search (optional)**: Perform exact search on deterministic cluster nodes and their neighbors to refine the graph locally. This phase is skipped by default due to its computational cost.

When no deterministic relations are present in the data, DGES behaves identically to standard GES.


Usage
----------------------------
.. code-block:: python

    from causallearn.search.ScoreBased.DGES import dges

    # Basic usage with default deterministic BIC score
    Record = dges(X)

    # With standard BIC score (for non-deterministic data)
    Record = dges(X, score_func='local_score_BIC')

    # With Phase 3 exact search enabled
    Record = dges(X, skip_exact_search=False)

    # With custom parameters
    Record = dges(X, score_func='local_score_BIC',
                  parameters={'lambda_value': 2.0},
                  node_names=['X1', 'X2', 'X3'])

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
and n_features is the number of features.

**score_func**: The score function to use. Default: ``'local_score_BIC_from_cov_deterministic'``.
              - ``'local_score_BIC_from_cov_deterministic'``: Deterministic-aware BIC score that handles zero residual variance (recommended for data with deterministic relations).
              - ``'local_score_BIC'``: Standard BIC score (use when no deterministic relations are present).
              - ``'local_score_BDeu'``: BDeu score.

**maxP**: Allowed maximum number of parents when searching the graph. Default: None (uses sample size N).

**parameters**: Additional parameters for the score function. Default: None.
              - parameters['lambda_value']: Penalty hyperparameter for BIC score. Default: 0.5.

**node_names**: list of str or None. Custom names for graph nodes. Default: None (uses ``X1, X2, ...``).

**det_threshold**: float. Threshold on eigenvalue ratio for detecting deterministic relations. Smaller values require stricter determinism. Default: ``1e-6``.

**det_epsilon**: float. Small constant added to residual variance in the deterministic BIC score to avoid log(0). Default: ``1e-8``.

**exact_search_method**: Method for Phase 3 exact search. Default: ``'dp'``.
              - ``'dp'``: Dynamic programming.
              - ``'astar'``: A* search.

**skip_exact_search**: bool. If True, skip Phase 3 (exact search) and return the result after Phase 2. Default: ``True``.


Returns
-------------------
- **Record['G']**: GeneralGraph. The learned causal graph (CPDAG), where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicate i --> j; Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.

- **Record['score']**: float. The score of the learned graph.

- **Record['update1']**: each update (Insert operator) in the forward step.

- **Record['update2']**: each update (Delete operator) in the backward step.

- **Record['G_step1']**: learned graph at each step in the forward step.

- **Record['G_step2']**: learned graph at each step in the backward step.

- **Record['mindcs']**: list of lists. Detected Minimal Deterministic Clusters. Each inner list contains the variable indices forming a MinDC.

- **Record['mindc_sets']**: list of sets. Same as mindcs but as sets.

- **Record['det_clusters']**: list of sets. Deterministic cluster groups (unions of overlapping MinDCs).

- **Record['exact_search_nodes']**: list. Nodes included in Phase 3 exact search (empty if skip_exact_search=True).

.. [1] Li, L., Dai, H., Al Ghothani, H., Huang, B., Zhang, J., Harel, S., ... & Zhang, K. (2024). On causal discovery in the presence of deterministic relations. Advances in Neural Information Processing Systems, 37, 130920-130952.
