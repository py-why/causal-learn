(Conditional) independence tests
===================
In this section, we would like to introduce (conditional) independence tests in causal-learn. Currently we have Fisher-z test [1]_,
Missing-value Fisher-z test, Chi-Square test, Kernel-based conditional independence (KCI) test and independence test [2]_,
and G-Square test [3]_.

(For more efficient nonparametric test, you may try `FastKCI and RCIT  <https://github.com/py-why/causal-learn/pull/202>`_. Both implementations are still preliminary and there might be some issues.)


Contents:

.. toctree::
    :titlesonly:
    :maxdepth: 2

    fisherz
    mvfisherz
    chisq
    kci
    gsq


.. [1] Fisher, R. A. (1921). On the'probable error'of a coefficient of correlation deduced from a small sample. Metron, 1, 1-32.
.. [2] Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011, July). Kernel-based Conditional Independence Test and Application in Causal Discovery. In 27th Conference on Uncertainty in Artificial Intelligence (UAI 2011) (pp. 804-813). AUAI Press.
.. [3] Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian network structure learning algorithm. Machine learning, 65(1), 31-78.
