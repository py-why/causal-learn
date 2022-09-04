.. _Kernel-based conditional independence (KCI) test and independence test:

Kernel-based conditional independence (KCI) test and independence test
=========================================================================

Kernel-based conditional independence (KCI) test and independence test [1]_.
To test if x and y are conditionally or unconditionally independent on Z. For unconditional independence tests,
Z is set to the empty set.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import CIT
    kci_obj = CIT(data, "kci") # construct a CIT instance with data and method name
    pValue = kci_obj(X, Y, S)

The above code runs KCI with the default parameters. Or instead if you would like to specify some parameters of KCI, you may do it by e.g.,

.. code-block:: python

    kci_obj = CIT(data, "kci", kernelZ='Polynomial', approx=False, est_width='median', ...)

See `KCI.py <https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/KCI/KCI.py>`_
for more details on the parameters options of the KCI tests.


Please be kindly informed that we have refactored the independence tests from functions to classes since the release `v0.1.2.8 <https://github.com/cmu-phil/causal-learn/releases/tag/0.1.2.8>`_. Speed gain and a more flexible parameters specification are enabled.

For users, you may need to adjust your codes accordingly. Specifically, if you are

+ running a constraint-based algorithm from end to end: then you don't need to change anything. Old codes are still compatible. For example,
.. code-block:: python

    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import kci
    cg = pc(data, 0.05, kci)

+ explicitly calculating the p-value of a test: then you need to declare the :code:`kci_obj` and then call it as above, instead of using :code:`kci(data, X, Y, condition_set)` as before. Note that now :code:`causallearn.utils.cit.kci` is a string :code:`"kci"`, instead of a function.

Please see `CIT.py <https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/cit.py>`_
for more details on the implementation of the (conditional) independent tests.

Parameters
------------
**data**: numpy.ndarray, shape (n_samples, n_features). Data, where n_samples is the number of samples
and n_features is the number of features.

**method**: string, "kci".

**kwargs**:

+ Either for specifying parameters of KCI, including:

  **KernelX/Y/Z (condition_set)**: ['GaussianKernel', 'LinearKernel', 'PolynomialKernel']. (For 'PolynomialKernel', the default degree is 2. Currently, users can change it by setting the 'degree' of 'class PolynomialKernel()'.

  **est_width**: set kernel width for Gaussian kernels.
   - 'empirical': set kernel width using empirical rules (default).
   - 'median': set kernel width using the median trick.

  **polyd**: polynomial kernel degrees (default=2).

  **kwidthx/y/z**: kernel width for data x/y/z (standard deviation sigma).

  **and more**: aee `KCI.py <https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/KCI/KCI.py>`_ for details.

+ Or for advanced usages of CIT, e.g., :code:`cache_path`. See :ref:`Advanced Usages <Advanced Usages>`.


Returns
-----------
**p**: the p value.


.. [1] Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011, July). Kernel-based Conditional Independence Test and Application in Causal Discovery. In 27th Conference on Uncertainty in Artificial Intelligence (UAI 2011) (pp. 804-813). AUAI Press.