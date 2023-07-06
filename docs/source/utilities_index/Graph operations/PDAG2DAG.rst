PDAG2DAG
============

Convert a PDAG to its corresponding DAG.

Usage
--------
.. code-block:: python

    from causallearn.utils.PDAG2DAG import pdag2dag
    Gd = pdag2dag(G)

Parameters
---------------
**G**: Partially Direct Acyclic Graph.

Returns
------------------
**Gd**: Direct Acyclic Graph.