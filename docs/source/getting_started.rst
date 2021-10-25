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

To help users get a quick sense of running time, we conducted benchmarking for several methods. We consider datasets with number of variables from {10, 25, 50, 100} and average degree from {2, 3, 4, 5}. The sample size is 1000. Results are as follows (more results will be updated soon).


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



