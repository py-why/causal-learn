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

To use Pytrad, we could install it using `pip <https://pypi.org/project/sqlparse/>`_:

.. code-block:: console

   (.venv) $ pip install pytrad


Install from source
------------

For development version, please kindly refer to our `GitHub Repository <https://github.com/cmu-phil/pytrad>`_.


Running examples
^^^^^^^^^^^^

For search methods in causal discovery, there are various running examples in the 'tests' directory in our `GitHub Repository <https://github.com/cmu-phil/pytrad>`_,
such as TestPC.py and TestGES.py.

For the implemented modules, such as (conditional) independent test methods, we provide unit tests for the convenience of developing your own methods.


Quick benchmarking
^^^^^^^^^^^^

To help users get a quick sense of running time, we conducted benchmarking for several methods. We consider datasets with number of variables from {10, 25, 50, 100} and average degree from {2, 3, 4, 5}. The sample size is 1000. We denote running time as 'S' if it is more than one day. All results are average over 10 runs with different random seeds.

.. list-table:: Running time (s) for PC (Fisher-z test)
   :widths: 20, 20, 20, 20, 20
   :header-rows: 1

   * -
     - 10
     - 25
     - 50
     - 100
   * - 2
     - 0.4
     - 4.5
     - 17.6
     - 58.7
   * - 3
     - 0.7
     - 5.9
     - 35.2
     - 79.9
   * - 4
     - 1.2
     - 12.4
     - 62.3
     - 122.7
   * - 5
     - 2
     - 18.4
     - 94.5
     - 217.8


.. list-table:: Running time (s) for FCI (Fisher-z test)
   :widths: 20, 20, 20
   :header-rows: 1

   * -
     - 10
     - 25

   * - 2
     - 1.9
     - 18968.5

   * - 3
     - 4.55
     - S

   * - 4
     - 7.65
     - S

   * - 5
     - 7.24
     - S

.. list-table:: Running time (s) for GES (BIC score)
   :widths: 20, 20, 20, 20
   :header-rows: 1

   * -
     - 10
     - 25
     - 50

   * - 2
     - 5.3
     - 170.4
     - 2384.8

   * - 3
     - 7.1
     - 296.1
     - 5534.5

   * - 4
     - 9.1
     - 2392.6
     - 9060.9

   * - 5
     - 13.5
     - 1368.7
     - 16323.5


Contributors
^^^^^^^^^^^^

**Group Leaders**: Kun Zhang, Joseph Ramsey, Shohei Shimizu, Peter Spirtes, Clark Glymour

**Coordinators**: Yujia Zheng, Mingming Gong, Biwei Huang, Wei Chen

**Contributors**:

Wei Chen, Ruichu Cai, Biwei Huang, Yuequn Liu, Zhiyi Huang: :ref:`PC <pc>`, :ref:`FCI <fci>`, :ref:`GES <ges>`, :ref:`GIN <gin>`, and :ref:`graph operaitions <graphoperation>`.

Mingming Gong, Erdun Gao: :ref:`PNL <pnl>`, :ref:`ANM <anm>`, :ref:`Granger causality <granger>`, and :ref:`KCI <Kernel-based conditional independence (KCI) test and independence test>`.

Shohei Shimizu, Takashi Nicholas Maeda, Takashi Ikeuchi: :ref:`LiNGAM-based methods <lingam>`.

Madelyn Glymour: several helpers.

Ruibo Tu: :ref:`Missing-value/test-wise deletion PC <pc>`.

Wai-Yin Lam: :ref:`PC <pc>`.

Biwei Huang: :ref:`CD-NOD <cdnod>`.

Ignavier Ng, Yujia Zheng: :ref:`Exact search <exactsearch>`.

Joseph Ramsey, Wei Chen, Zhiyi Huang: :ref:`Evaluations <evaluation>`.



