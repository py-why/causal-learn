.. _Generalized score with cross validation:

Generalized score with cross validation
=============================================

Generalized score with cross validation for single-dimensional variables
-----------------------------------------------------------------------------
Calculate the local score using negative k-fold cross-validated log likelihood as the score, based on a regression model in RKHS [1]_.


Usage
^^^^^^^
.. code-block:: python

    from causallearn.score.LocalScoreFunction import local_score_cv_general
    score = local_score_cv_general(Data, Xi, PAi, parameters)

Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Data**: (sample, features).

**Xi**: current index.

**PAi**: parent indexes.

**parameters**:
               - kfold: the fold number in cross validation.
               - lambda: regularization parameter.

Returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**score**: Local score.


Generalized score with cross validation for multi-dimensional variables
------------------------------------------------------------------------------
Calculate the local score using negative k-fold cross-validated log likelihood as the score,
based on a regression model in RKHS for data with multi-dimensional variables [1]_.

Usage
^^^^^^^
.. code-block:: python

    from causallearn.score.LocalScoreFunction import local_score_cv_multi
    score = local_score_cv_multi(Data, Xi, PAi, parameters)


Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Data**: (sample, features).

**Xi**: current index.

**PAi**: parent indexes.

**parameters**:
               - kfold: the fold number in cross validation.
               - lambda: regularization parameter.
               - dlabel: indicate the data dimensions that belong to each variable. It is only used when the variables have multivariate dimensions.

Returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**score**: Local score.

.. [1] Huang, B., Zhang, K., Lin, Y., Schölkopf, B., & Glymour, C. (2018, July). Generalized score functions for causal discovery. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 1551-1560).
