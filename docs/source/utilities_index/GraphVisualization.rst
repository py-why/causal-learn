.. _graph_visualization:

Graph Visualization
==============================================

``GraphUtils.plot_graph()`` renders causal graphs with optional colored nodes. It supports
manual per-node coloring, automatic category-based coloring, custom labels, titles, and
saving to file.


Usage
----------------------------

Basic Plot
^^^^^^^^^^
.. code-block:: python

    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.GraphUtils import GraphUtils

    # Run GES to get a graph
    Record = ges(X)

    # Simple plot (no colors)
    GraphUtils.plot_graph(Record['G'])

Plot with Custom Node Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    GraphUtils.plot_graph(Record['G'],
                          labels=['Age', 'Income', 'Spending', 'Education'])

Plot with Manual Node Colors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # Assign a specific color to each node by name
    colors = {
        'Age': 'lightblue',
        'Income': 'lightcoral',
        'Spending': 'lightgreen',
        'Education': 'lightsalmon',
    }
    GraphUtils.plot_graph(Record['G'],
                          labels=['Age', 'Income', 'Spending', 'Education'],
                          colors=colors,
                          title='Causal Graph with Manual Colors',
                          save_path='my_graph.png')

Plot with Category-based Colors (Auto-assigned)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # Group features into categories — each category gets a unique color
    # Node labels are auto-derived from the category dict (flattened in order)
    category_to_features = {
        'demographics': ['Age', 'Education'],
        'financial': ['Income', 'Spending'],
    }
    GraphUtils.plot_graph(Record['G'],
                          category_to_features=category_to_features,
                          title='Causal Graph by Category',
                          save_path='graph_by_category.png')

Full Example: GES + Colored Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    import numpy as np
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.GraphUtils import GraphUtils

    # Generate sample data
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5)
    X[:, 1] = X[:, 0] * 0.8 + np.random.randn(n) * 0.5
    X[:, 2] = X[:, 1] * 0.6 + np.random.randn(n) * 0.4

    # Run GES
    Record = ges(X, node_names=['Drug_A', 'Drug_B', 'Recovery',
                                'Side_Effect', 'Dosage'])

    # Visualize with category colors
    categories = {
        'treatment': ['Drug_A', 'Drug_B', 'Dosage'],
        'outcome': ['Recovery', 'Side_Effect'],
    }
    GraphUtils.plot_graph(Record['G'],
                          category_to_features=categories,
                          title='Drug Study Causal Graph',
                          save_path='drug_study.png',
                          dpi=300)


Parameters
-------------------

**G**: Graph. A causal-learn graph object (e.g., output of ``ges()`` or ``dges()``).

**labels**: list of str or None. Node labels. Must have the same length as ``G.get_nodes()``. Default: None (uses graph node names).

**colors**: dict or None. Mapping from node label (str) to fill color (str). Any valid CSS color name can be used (e.g., ``'lightblue'``, ``'#FF6B6B'``). If both ``colors`` and ``category_to_features`` are provided, ``colors`` takes precedence. Default: None.

**category_to_features**: dict or None. Mapping from category name (str) to list of feature names (list of str). Each category is automatically assigned a distinct color from a built-in palette. If ``labels`` is not provided, node labels are auto-derived by flattening this dict in order. Default: None.

**save_path**: str or None. File path to save the rendered image (e.g., ``'output/graph.png'``). Supports PNG, PDF, SVG formats. If None, the image is only displayed. Default: None.

**title**: str. Title displayed above the graph. Default: ``""``.

**dpi**: float. Resolution in dots per inch. Default: 500.

**figsize**: tuple. Figure size in inches (width, height). Default: (20, 12).


get_category_colors()
----------------------------

A helper function that generates a color mapping from a category dictionary.
Used internally by ``plot_graph()``, but can also be called directly if you need
to customize the color mapping before plotting.

.. code-block:: python

    from causallearn.utils.GraphUtils import GraphUtils

    categories = {
        'treatment': ['Drug_A', 'Drug_B'],
        'outcome': ['Recovery', 'Side_Effect'],
    }
    colors = GraphUtils.get_category_colors(categories)
    # Result: {'Drug_A': 'lightblue', 'Drug_B': 'lightblue',
    #          'Recovery': 'lightcoral', 'Side_Effect': 'lightcoral'}

    # Customize one color, then plot
    colors['Drug_A'] = 'gold'
    GraphUtils.plot_graph(G, colors=colors)

**Parameters**:

- **category_to_features**: dict. Mapping from category name to list of feature names.

**Returns**:

- **feature_to_color**: dict. Mapping from each feature name (str) to a CSS color string. All features in the same category share the same color.


Notes
----------------------------
- **Dependencies**: Requires ``matplotlib`` and ``pydot`` (which requires Graphviz to be installed).
- **Edge notation**: Directed edges are shown as arrows. Undirected edges are shown as lines.
- **Color priority**: If both ``colors`` and ``category_to_features`` are passed, ``colors`` is used and ``category_to_features`` is ignored.
- **Nodes without colors**: If a node label is not found in the ``colors`` dict, it will be rendered with the default white background.
