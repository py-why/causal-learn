.. _Custom CI tests:

Custom CI tests
===================================

In addition to the built-in tests above, ``causallearn.utils.cit.CIT`` supports
registering your own — or a third-party library's — (conditional) independence
test implementation, so it can be selected by name just like any built-in
method — including from constraint-based algorithms such as :code:`pc`,
:code:`fci` and :code:`cdnod`.

Usage
--------
.. code-block:: python

    from causallearn.utils.cit import CIT_Base, register_ci_test

    class MyCITest(CIT_Base):
        def __init__(self, data, **kwargs):
            super().__init__(data, **kwargs)
            self.method = "my_test"

        def __call__(self, X, Y, condition_set=None):
            # ... compute and return a single p-value (float) ...
            return p_value

    register_ci_test("my_test", MyCITest)

    # now usable anywhere a built-in method name is accepted
    from causallearn.utils.cit import CIT
    from causallearn.search.ConstraintBased.PC import pc

    cit = CIT(data, "my_test")
    pValue = cit(X, Y, S)

    cg = pc(data, indep_test="my_test")

Please see `CIT.py <https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/cit.py>`_
for more details on the implementation of the (conditional) independence tests.

Parameters
------------
**test_class**: a class subclassing :code:`CIT_Base` and implementing
:code:`__call__(self, X, Y, condition_set=None)`, returning a single p-value (float).

**name**: string, the name under which the test is registered — pass it as
:code:`method` to :code:`CIT()`, or as :code:`indep_test` /
:code:`independence_test_method` to :code:`pc` / :code:`fci` / :code:`cdnod`.

**kwargs**: any constructor arguments your test class accepts are passed through,
e.g. :code:`CIT(data, "my_test", **kwargs)`.

Returns
-----------
**p**: the p-value of the test, same as any built-in method.

Example: GPU-accelerated CI tests from causal-ts
--------------------------------------------------

`causal-ts <https://github.com/bloomberg/causal-ts>`_, a companion library for causal
discovery in (nonstationary) time series, ships a set of GPU-accelerated CI tests
that register themselves with causal-learn through this exact mechanism:

+ ``kci_gpu`` — kernel-based CI
+ ``splitkci_gpu`` — bias-controlled, split-sample kernel CI
+ ``dfcit_gpu`` — distribution-free CI, handles mixed discrete/continuous data
+ ``rcot_gpu`` — random Fourier features CI
+ ``cmiknn_gpu`` — conditional mutual information (k-NN)
+ ``gcmi_gpu`` — Gaussian-copula mutual information
+ \+more (see the source below for the full, up-to-date list of registered names)

.. code-block:: python

    # pip install causalts
    import causalts.ci_tests.causal_learn_bridge  # registers the tests above on import

    from causallearn.utils.cit import CIT
    from causallearn.search.ConstraintBased.PC import pc

    cit = CIT(data, method="kci_gpu", device="cuda")
    pValue = cit(X, Y, S)

    cg = pc(data, indep_test="kci_gpu", device="cuda")

See `causal_learn_bridge.py
<https://github.com/bloomberg/causal-ts/blob/main/causalts/ci_tests/causal_learn_bridge.py>`_
for the source of these adapters.

These pay off most for tests whose cost scales poorly with sample size on
CPU. causal-learn's own ``kci`` is :math:`O(n^3)` per call; ``kci_gpu``
computes the same kernel-based statistic on the GPU and scales far more
gently. Measured with ``pc()`` on an NVIDIA L4 (single run, your numbers
will vary by hardware and data):

.. code-block:: python

    import time
    from causallearn.search.ConstraintBased.PC import pc

    t0 = time.time()
    cg_cpu = pc(data, indep_test="kci", show_progress=False)
    t1 = time.time()
    cg_gpu = pc(data, indep_test="kci_gpu", device="cuda", show_progress=False)
    t2 = time.time()

    print(f"kci:      {t1 - t0:.2f}s")
    print(f"kci_gpu:  {t2 - t1:.2f}s")
    print("same graph:", bool((cg_cpu.G.graph == cg_gpu.G.graph).all()))

    # n=1000, d=20 (NVIDIA L4): kci=1.13s, kci_gpu=0.09s, same graph: True

The speedup grows with sample size (0.9x at n=200 up to ~19x at n=1500 in a
direct single-call comparison), since ``kci``'s cubic cost keeps growing
while ``kci_gpu`` does not. Pick the test that matches your data (linear vs.
nonlinear), not by device alone — for a simple linear-Gaussian test,
causal-learn's own ``fisherz`` precomputes a single correlation matrix and is
already near-instant, which no per-call GPU dispatch beats.
