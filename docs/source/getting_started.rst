=============
Getting started
=============


Installation
^^^^^^^^^^^^

Requirements
------------

* python 3
* numpy
* networkx
* pandas
* scipy
* scikit-learn
* statsmodels
* pydot

(For visualization)

* matplotlib
* graphviz
* pygraphviz (might not support the most recent Mac)


Install via PyPI
------------

To use causal-learn, we could install it using `pip <https://pypi.org/project/sqlparse/>`_:

.. code-block:: console

   (.venv) $ pip install causal-learn


Install from source
------------

For development version, please kindly refer to our `GitHub Repository <https://github.com/cmu-phil/causal-learn>`_.


Running examples
^^^^^^^^^^^^

For search methods in causal discovery, there are various running examples in the 'tests' directory in our `GitHub Repository <https://github.com/cmu-phil/causal-learn>`_,
such as TestPC.py and TestGES.py.

For the implemented modules, such as (conditional) independent test methods, we provide unit tests for the convenience of developing your own methods.

Benchmarks
^^^^^^^^^^^^
For the convenience of our community, `CMU-CLeaR <https://www.cmu.edu/dietrich/causality>`_ group maintains a list of benchmark datasets including real-world scenarios and various learning tasks. Please refer to the following links:

- `https://github.com/cmu-phil/example-causal-datasets <https://github.com/cmu-phil/example-causal-datasets>`_ (maintained by Joseph Ramsey)
- `https://www.cmu.edu/dietrich/causality/causal-learn <https://www.cmu.edu/dietrich/causality/causal-learn>`_

Please feel free to let us know if you have any recommendation regarding causal datasets with high-quality. We are grateful for any effort that benefits the development of causality community.

Contributors
^^^^^^^^^^^^

**Team Leaders**: Kun Zhang, Joseph Ramsey, Mingming Gong, Ruichu Cai, Shohei Shimizu, Peter Spirtes, Clark Glymour

**Coordinators**: Yujia Zheng, Biwei Huang, Wei Chen

**Developers**:

- Wei Chen, Biwei Huang, Yuequn Liu, Zhiyi Huang, Feng Xie, Haoyue Dai: :ref:`PC <pc>`, :ref:`FCI <fci>`, :ref:`GES <ges>`, :ref:`GIN <gin>`, and :ref:`graph operations <graphoperation>`.
- Mingming Gong, Erdun Gao: :ref:`PNL <pnl>`, :ref:`ANM <anm>`, :ref:`Granger causality <granger>`, and :ref:`KCI <Kernel-based conditional independence (KCI) test and independence test>`.
- Shohei Shimizu, Takashi Nicholas Maeda, Takashi Ikeuchi: :ref:`LiNGAM-based methods <lingam>`.
- Madelyn Glymour: several helpers.
- Ruibo Tu: :ref:`Missing-value/test-wise deletion PC <pc>`.
- Wai-Yin Lam: :ref:`PC <pc>`.
- Biwei Huang: :ref:`CD-NOD <cdnod>`.
- Ignavier Ng, Yujia Zheng: :ref:`Exact search <exactsearch>`.
- Joseph Ramsey, Wei Chen, Zhiyi Huang: :ref:`Evaluations <evaluation>`.



