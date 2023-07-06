DAG2CPDAG
==============

Convert a DAG to its corresponding CPDAG.

Usage
--------
.. code-block:: python

    from causallearn.utils.DAG2CPDAG import dag2cpdag
    CPDAG = dag2cpdag(G)

Parameters
---------------------
**G**: Direct Acyclic Graph.

Returns
--------------
**CPDAG**: Completed Partially Direct Acyclic Graph.
