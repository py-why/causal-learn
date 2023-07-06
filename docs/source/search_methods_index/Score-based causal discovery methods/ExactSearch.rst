.. _exactsearch:

Exact Search
=====================

Algorithm Introduction
--------------------------------------
Search for the optimal graph using Dynamic Programming (DP [1]_) or A* search [2]_.

Usage
--------------------------------------
.. code-block:: python

    from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
    dag_est, search_stats = bic_exact_search(X, super_graph, search_method,
                     use_path_extension, use_k_cycle_heuristic,
                     k, verbose, include_graph, max_parents)

Parameters
--------------------------------------
**X**: numpy.ndarray, shape=(n, d).
The data to fit the structure too, where each row is a sample and
each column corresponds to the associated variable.

**super_graph**: numpy.ndarray, shape=(d, d).
Super-structure to restrict search space (binary matrix).
If None, no super-structure is used. Default is None.

**search_method**: str.
Method of exact search (['astar', 'dp']).
Default is astar.

**use_path_extension**: bool.
Whether to use optimal path extension for order graph. Note that
this trick will not affect the correctness of search procedure.
Default is True.

**use_k_cycle_heuristic**: bool.
Whether to use k-cycle conflict heuristic for astar.
Default is False.

**k**: int.
Parameter used by k-cycle conflict heuristic for astar.
Default is 3.

**verbose**: bool.
Whether to log messages related to search procedure.

**max_parents**: int.
The maximum number of parents a node can have. If used, this means
using the k-learn procedure. Can drastically speed up algorithms.
If None, no max on parents. Default is None.

Returns
--------------------------------------
**dag_est**:  numpy.ndarray, shape=(d, d). Estimated DAG.

**search_stats**:  dict. Some statistics related to the search procedure.

.. [1] Silander, T., & Myllymäki, P. (2006, July). A simple approach for finding the globally optimal Bayesian network structure. In Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence (pp. 445-452).
.. [2] Yuan, C., & Malone, B. (2013). Learning optimal Bayesian networks: A shortest path perspective. Journal of Artificial Intelligence Research, 48, 23-65.
