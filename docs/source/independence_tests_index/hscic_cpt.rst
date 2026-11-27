.. _HSCIC with the conditional permutation test:

HSCIC with the conditional permutation test
============================================

HSCIC-CPT combines the Hilbert--Schmidt conditional independence criterion
(HSCIC) [1]_ with the conditional permutation test (CPT) [2]_. The combined
procedure and its training-only regularisation search are described in [3]_.

The test randomly divides the observations into training and evaluation
blocks. For each direction, the training block estimates the conditional law
of the permuted endpoint given ``Z`` and selects the corresponding HSCIC
regularisation parameter. The final directional p-values are computed on the
disjoint evaluation block. Random seeds are derived separately for every
query, so results are reproducible and do not depend on the order in which a
causal discovery algorithm requests CI tests.

CPT is directional: it models and permutes its first argument. Conditional
independence is symmetric, however, and a CI test used for causal discovery
should not depend on which endpoint is called ``X``. The ``CIT`` wrapper
therefore evaluates both directions, using conditional models for ``X | Z``
and ``Y | Z``, and returns

.. math::

    \min\{1,\;2\min(p_{X\mid Z},p_{Y\mid Z})\}.

This Bonferroni combination is valid without assuming that the two
directional p-values are independent. The lower-level ``HSCICCPT`` class
retains the one-direction procedure described in the original papers.

Usage
-----

.. code-block:: python

    from causallearn.utils.cit import CIT

    hscic_cpt_obj = CIT(data, "hscic_cpt", random_state=7)
    p_value = hscic_cpt_obj(X, Y, S)

The same test can be used in PC:

.. code-block:: python

    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import hscic_cpt

    cg = pc(
        data,
        0.05,
        hscic_cpt,
        random_state=7,
    )

PC passes its significance level to the HSCIC regularisation search
automatically.

Conditional model
-----------------

CPT requires a conditional density for the endpoint it permutes. By default,
``hscic_cpt`` estimates scalar linear-Gaussian models for ``X | Z`` and
``Y | Z`` on the training block. This matches the Gaussian helper in the CPT
authors' reference implementation. It is appropriate only when these
conditional models are reasonable approximations.

For another conditional model, pass a callable as ``conditional_density``.
It receives ``(target_train, z_train, target_test, z_test)`` and must return
two square matrices. Entry ``[i, j]`` of each matrix must equal
``log qhat(target_i | z_j)`` up to an additive term that depends only on
``z_j``. The callback must fit or otherwise select its model using the
training arguments only. For a conditional query the wrapper calls it once
with ``target=X`` and once with ``target=Y``.

.. code-block:: python

    def conditional_density(x_train, z_train, x_test, z_test):
        fitted_model = fit_model(x_train, z_train)
        return (
            fitted_model.log_density_matrix(x_train, z_train),
            fitted_model.log_density_matrix(x_test, z_test),
        )

    hscic_cpt_obj = CIT(
        data,
        "hscic_cpt",
        conditional_density=conditional_density,
        random_state=7,
    )

Approximate validity depends on the quality of this conditional model. HSCIC-
CPT is not a distribution-free conditional independence test when
``qhat(X | Z)`` is estimated inaccurately.

Parameters
----------

**data**: numpy.ndarray, shape (n_samples, n_features)
    The data matrix.

**train_fraction**: float, default=0.5
    Fraction of observations reserved for conditional-model fitting and
    HSCIC regularisation selection.

**hscic_alpha**: float, default=0.05
    Rejection rate targeted by the training-only regularisation search. Set
    this to the significance level used by the surrounding causal discovery
    algorithm.

**ridge_grid**: sequence of positive floats
    Candidate HSCIC regularisation parameters. The default contains 18
    log-spaced values between ``1e-4`` and ``1e-1``.

**n_pretests**: int, default=100
    Number of null-true training permutations used to estimate rejection
    rates during regularisation selection.

**n_reference_permutations**: int, default=199
    Number of reference permutations used for each training pretest.

**n_permutations**: int, default=500
    Number of conditional permutations in the final evaluation test.

**n_steps**: int, default=50
    Number of parallel pairwise-swap steps in each CPT Markov-chain branch.

**n_landmarks**: int or None, default=None
    Number of conditioning observations over which HSCIC is averaged.
    ``None`` uses every evaluation observation.

**random_state**: int, default=0
    Seed controlling the split, landmark selection and conditional
    permutations.

Returns
-------

**p**: float
    The conditional permutation p-value.


.. [1] Park, J., & Muandet, K. (2020). `A Measure-Theoretic Approach to
   Kernel Conditional Mean Embeddings
   <https://proceedings.neurips.cc/paper/2020/hash/f340f1b1f65b6df5b5e3f94d95b11daf-Abstract.html>`_.
   Advances in Neural Information Processing Systems 33.
.. [2] Berrett, T. B., Wang, Y., Barber, R. F., & Samworth, R. J. (2020).
   `The Conditional Permutation Test for Independence While Controlling for
   Confounders <https://academic.oup.com/jrsssb/article/82/1/175/7056014>`_.
   Journal of the Royal Statistical Society: Series B, 82(1), 175--197.
.. [3] Laumann, F., von Kügelgen, J., Park, J., Schölkopf, B., & Barahona, M.
   (2023). `Kernel-Based Independence Tests for Causal Structure Learning on
   Functional Data <https://www.mdpi.com/1099-4300/25/12/1597>`_. Entropy,
   25(12), 1597.
