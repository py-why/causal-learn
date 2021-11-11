# causal-learn: Causal Discovery for Python

Causal-learn is a python package for causal discovery that implements both classical and state-of-the-art causal discovery algorithms, which is a Python translation and extension of [Tetrad](https://github.com/cmu-phil/tetrad).

The package is actively being developed. Feedbacks (issues, suggestions, etc.) are highly encouraged.

# Package Overview

Our causal-learn implements methods for causal discovery:

* Constrained-based causal discovery methods.
* Score-based causal discovery methods.
* Causal discovery methods based on constrained functional causal models.
* Hidden causal representation learning.
* Granger causality.
* Multiple utilities for building your own method, such as independence tests, score functions, graph operations, and evaluations.

# Install

Causal-learn needs the following packages to be installed beforehand:

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

To use causal-learn, we could install it using [pip](https://pypi.org/project/sqlparse/):

```
pip install causal-learn
```

# Documentation

Please kindly refer to [causal-learn Doc](https://causal-learn.readthedocs.io/en/latest/) for detailed tutorials and usages.
