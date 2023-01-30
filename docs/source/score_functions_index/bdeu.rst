.. _BDeu score:

BDeu score
==================
Calculate the local score with BDeu [1]_ for the discrete case.

Usage
--------
.. code-block:: python

    from causallearn.score.LocalScoreFunction import local_score_BDeu
    score = local_score_BDeu(Data, i, PAi, parameters)


Parameters
---------------
**Data**: (sample, features).

**i**: current index.

**PAi**: parent indexes.

**parameters**:
               - sample_prior: sample prior.
               - structure_prior: structure prior.
               - r_i_map: number of states of the finite random variable 'X_{i}'.

Returns
-----------------
**score**: Local BDeu score.

.. [1] Buntine, W. (1991). Theory refinement on Bayesian networks. In Uncertainty proceedings 1991 (pp. 52-60). Morgan Kaufmann.