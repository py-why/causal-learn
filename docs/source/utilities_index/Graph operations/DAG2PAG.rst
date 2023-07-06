DAG2PAG
==================

Convert a DAG to its corresponding PAG.

Usage
--------
.. code-block:: python

    from causallearn.utils.DAG2PAG import dag2pag
    PAG = dag2pag(dag, islatent)

Parameters
------------------------
**dag**: Direct Acyclic Graph.

**islatent**: the indexes of latent variables. [] means there is no latent variable.

Returns
--------------------
**PAG**: Partial Ancestral Graph.