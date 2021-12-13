.. _fci:

FCI
=====

Algorithm Introduction
--------------------------------------

Causal Discovery with Fast Causal Inference (FCI [1]_).


Usage
----------------------------
.. code-block:: python

    from causallearn.search.ConstraintBased.FCI import fci
    G = fci(data, indep_test, alpha, verbose, background_knowledge)

    # visualization
    pgv_g = GraphUtils.to_pgv(G)
    pgv_g.draw('simple_test.png', prog='dot', format='png')
Parameters
-------------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**alpha**: Significance level of individual partial correlation tests.

**indep_test**: Independence test method function.
       - ":ref:`fisherz <Fisher-z test>`": Fisher's Z conditional independence test.
       - ":ref:`chisq <Chi-Square test>`": Chi-squared conditional independence test.
       - ":ref:`gsq <G-Square test>`": G-squared conditional independence test.
       - ":ref:`kci <Kernel-based conditional independence (KCI) test and independence test>`": kernel-based conditional independence test. (As a kernel method, its complexity is cubic in the sample size, so it might be slow if the same size is not small.)
       - ":ref:`mv_fisherz <Missing-value Fisher-z test>`": Missing-value Fisher's Z conditional independence test.

**verbose**: 0 - no output, 1 - detailed output.

**background_knowledge**: class BackgroundKnowledge. Add prior edges according to assigned causal connections.
For detailed usage, please kindly refer to its `usage example <https://github.com/cmu-phil/causal-learn/blob/main/tests/TestBackgroundKnowledge.py>`_.



Returns
-------------------
**G** : a GeneralGraph object. Nodes in the graph correspond to the column indices in the data. For visualization, please refer to the `running example <https://github.com/cmu-phil/causal-learn/tree/main/tests>`_.

.. [1] Spirtes, P., Meek, C., & Richardson, T. (1995, August). Causal inference in the presence of latent variables and selection bias. In Proceedings of the Eleventh conference on Uncertainty in artificial intelligence (pp. 499-506).
