TXT2GeneralGraph
==========================

Convert text file of Tetrad results into GeneralGraph class for causal-learn.

Usage
--------
.. code-block:: python

    from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
    G = txt2generalgraph(filename)

Parameters
-----------------
**filename**: Text file of Tetrad results.

Returns
------------------
**G**: GeneralGraph Class for causal-learn.