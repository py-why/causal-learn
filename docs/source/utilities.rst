=============
Utilities
=============

Graph operations
--------------------------------------------------

DAG2CPDAG
^^^^^^^^^^^^

Covert a DAG to its corresponding PDAG.

Parameters
""""""""""""""""""""""""""""""""""""
**G**: Direct Acyclic Graph.

Returns
""""""""""""""""""""""""""""""""""""
**CPDAG**: Completed Partially Direct Acyclic Graph.


DAG2PAG
^^^^^^^^^^^^

Covert a DAG to its corresponding PAG.

Parameters
""""""""""""""""""""""""""""""""""""
**dag**: Direct Acyclic Graph.

**islatent**: the indexes of latent variables. [] means there is no latent variable.

Returns
""""""""""""""""""""""""""""""""""""
**PAG**: Partial Ancestral Graph.


PDAG2DAG
^^^^^^^^^^^^

Covert a PDAG to its corresponding DAG.

Parameters
""""""""""""""""""""""""""""""""""""
**G**: Partially Direct Acyclic Graph.

Returns
""""""""""""""""""""""""""""""""""""
**Gd**: Direct Acyclic Graph.


TXT2GenralGraph
^^^^^^^^^^^^^^^^^^^^^^^^

Convert text file of Tetrad results into GeneralGraph class for causal-learn.

Parameters
""""""""""""""""""""""""""""""""""""
**filename**: Text file of Tetrad results.

Returns
""""""""""""""""""""""""""""""""""""
**G**: GeneralGraph Class for causal-learn.
