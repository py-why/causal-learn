.. _Kernel-based conditional independence (KCI) test and independence test:

Kernel-based conditional independence (KCI) test and independence test
=========================================================================

Kernel-based conditional independence (KCI) test and independence test [1]_.
To test if x and y are conditionally or unconditionally independent on Z. For unconditional independence tests,
Z is set to the empty set.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import kci
    p = kci(data, X, Y, condition_set, kernelX, kernelY, kernelZ, est_width, polyd, kwidthx, kwidthy, kwidthz)

Parameters
-------------
**X, Y, and condition_set**: data matrices of size number_of_samples * dimensionality. Z could be None.

**KernelX/Y/Z (condition_set)**: ['GaussianKernel', 'LinearKernel', 'PolynomialKernel'].
(For 'PolynomialKernel', the default degree is 2. Currently, users can change it by setting the 'degree' of 'class PolynomialKernel()'.

**est_width**: set kernel width for Gaussian kernels.
   - 'empirical': set kernel width using empirical rules (default).
   - 'median': set kernel width using the median trick.

**polyd**: polynomial kernel degrees (default=2).

**kwidthx**: kernel width for data x (standard deviation sigma).

**kwidthy**: kernel width for data y (standard deviation sigma).

**kwidthz**: kernel width for data z (standard deviation sigma).

Returns
-----------
**p**: the p value.


.. [1] Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011, July). Kernel-based Conditional Independence Test and Application in Causal Discovery. In 27th Conference on Uncertainty in Artificial Intelligence (UAI 2011) (pp. 804-813). AUAI Press.