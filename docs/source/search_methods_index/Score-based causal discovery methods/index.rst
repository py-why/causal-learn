Score-based causal discovery methods
================================================
In this section, we would like to introduce Score-based causal discovery methods, including GES, DGES, and Exact search methods.
For GES, we implemented it with BIC score [1]_ and generalized score [2]_. DGES [5]_ extends GES to handle data with deterministic (functional) relationships among variables. For Exact search, we implemented DP [3]_ and A* [4]_.


Contents:

.. toctree::
    :maxdepth: 2

    GES
    DGES
    ExactSearch

.. [1] Chickering, D. M. (2002). Optimal structure identification with greedy search. Journal of machine learning research, 3(Nov), 507-554.
.. [2] Huang, B., Zhang, K., Lin, Y., Schölkopf, B., & Glymour, C. (2018, July). Generalized score functions for causal discovery. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 1551-1560).
.. [3] Silander, T., & Myllymäki, P. (2006, July). A simple approach for finding the globally optimal Bayesian network structure. In Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence (pp. 445-452).
.. [4] Yuan, C., & Malone, B. (2013). Learning optimal Bayesian networks: A shortest path perspective. Journal of Artificial Intelligence Research, 48, 23-65.
.. [5] Li, L., Dai, H., Al Ghothani, H., Huang, B., Zhang, J., Harel, S., ... & Zhang, K. (2024). On causal discovery in the presence of deterministic relations. Advances in Neural Information Processing Systems, 37, 130920-130952.
