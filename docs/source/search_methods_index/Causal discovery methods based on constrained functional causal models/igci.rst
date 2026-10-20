.. _igci:

Information-Geometric Causal Inference (IGCI)
==============================================

Algorithm Introduction
----------------------

Information-Geometric Causal Inference (IGCI [1]_ [2]_) is a bivariate causal discovery
method designed for continuous variables where the relation is (near-)deterministic and invertible:

.. math::

    Y = f(X)

Unlike Additive Noise Models (ANM) or Post-Nonlinear models (PNL), IGCI does not rely on an
additive independent noise term. Instead, it relies on an information-geometric postulate:
the probability density of the cause :math:`X` is independent of the log-derivative of the
causal mechanism :math:`\log |f'|` relative to a reference measure :math:`\mu`.

Under this postulate, the forward model yields a lower geometric complexity score than the
backward model:

.. math::

    C_{X \to Y} < C_{Y \to X} \implies \text{infer } X \to Y

Reference Measures and Preprocessing
------------------------------------

The choice of reference measure :math:`\mu` is essential to the definition and performance of IGCI:

* **'uniform'** (default): Rescales each variable to the unit interval :math:`[0, 1]` via min-max normalisation.
  This reference measure is best suited when variables have bounded/compact support or approximate uniform marginals.
* **'gaussian'**: Standardises each variable to zero mean and unit variance (:math:`\mu=0, \sigma=1`).
  This reference measure is appropriate when variables are unbounded or approximately Gaussian.

**Preprocessing requirements**: Inputs must be continuous scalar variables with non-zero variance. Constant
features with zero range or zero variance cannot be normalised and will raise a ``ValueError``.

Estimators
----------

Two estimators are supported:

* **'slope'** (default): Evaluates the slope difference between consecutive sorted observations:

  .. math::

      \hat{C}_{X \to Y} = \frac{1}{m-1} \sum_{i=1}^{m-1} \log \left| \frac{\tilde{y}_{(i+1)} - \tilde{y}_{(i)}}{\tilde{x}_{(i+1)} - \tilde{x}_{(i)}} \right|

  Pairs with non-positive differences (ties) are skipped.

* **'entropy'**: Evaluates differential entropy difference :math:`C_{X \to Y} = H(\tilde{Y}) - H(\tilde{X})`
  using a 1-spacing nearest-neighbour entropy estimator with digamma correction (Janzing et al., 2012 [2]_).

Usage
-----

.. code-block:: python

    from causallearn.search.FCMBased.IGCI.IGCI import IGCI

    model = IGCI(ref_measure='uniform', estimator='slope')

    # Compute directional costs in both directions
    score_forward, score_backward = model.cause_or_effect(data_x, data_y)

    # Inferred direction: +1 for X -> Y, -1 for Y -> X, 0 for undecided
    direction = model.predict_direction(data_x, data_y)

    # Signed confidence statistic (score_backward - score_forward)
    confidence = model.compute_score(data_x, data_y)

Or using the functional interface:

.. code-block:: python

    from causallearn.search.FCMBased.IGCI.IGCI import igci

    score_forward, score_backward = igci(data_x, data_y, ref_measure='uniform', estimator='slope')

Parameters
----------

**ref_measure** : ``{'uniform', 'gaussian'}``, default=``'uniform'``
    Reference measure for variable normalisation.

**estimator** : ``{'slope', 'entropy'}``, default=``'slope'``
    Cost estimation method.

**data_x** : array-like of shape ``(n,)`` or ``(n, 1)``
    First continuous variable.

**data_y** : array-like of shape ``(n,)`` or ``(n, 1)``
    Second continuous variable.

Returns
-------

**score_forward** : float
    Cost :math:`C_{X \to Y}`. Smaller value indicates a simpler mechanism in the :math:`X \to Y` direction.

**score_backward** : float
    Cost :math:`C_{Y \to X}`. Smaller value indicates a simpler mechanism in the :math:`Y \to X` direction.

References
----------

.. [1] Daniusis, P., Janzing, D., Mooij, J., Zscheischler, J., Steudel, B., Zhang, K., & Schölkopf, B. (2010). Inferring deterministic causal relations. In Proceedings of the 26th Conference on Uncertainty in Artificial Intelligence (UAI 2010), pp. 143-150.
.. [2] Janzing, D., Mooij, J., Zhang, K., Lemeire, J., Zscheischler, J., Daniušis, P., Steudel, B., & Schölkopf, B. (2012). Information-geometric approach to inferring causal directions. Artificial Intelligence, 182-183, pp. 1-31.
.. [3] Zhang, S., Wu, J., Li, Z., Liu, L., Liao, J., & Sun, L. (2022). FIGCI: flow-based information-geometric causal inference. In Proceedings of the Chinese Conference on Artificial Intelligence (CICAI 2022), LNCS 13605, pp. 520-531.
