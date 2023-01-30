.. _BIC score:

BIC score
==========
Calculate the local score with Bayesian Information Criterion (BIC [1]_) for the linear Gaussian case.

Usage
--------
.. code-block:: python

    from causallearn.score.LocalScoreFunction import local_score_BIC
    score = local_score_BIC(Data, i, PAi, parameters)


Parameters
--------------------
**Data**: (sample, features).

**i**: current index.

**PAi**: parent indexes.

**parameters**: None.

Returns
-----------------
**score**: Local BIC score.

.. [1] Schwarz, G. (1978). Estimating the dimension of a model. The annals of statistics, 461-464.
