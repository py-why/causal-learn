# causal-learn: Causal Discovery for Python

Causal-learn is a python package for causal discovery that implements both classical and state-of-the-art causal discovery algorithms, which is a Python translation and extension of [Tetrad](https://github.com/cmu-phil/tetrad).

The package is actively being developed. Feedbacks (issues, suggestions, etc.) are highly encouraged.

# Package Overview

Our causal-learn implements methods for causal discovery:

* Constraint-based causal discovery methods.
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

# Running examples

For search methods in causal discovery, there are various running examples in the **‘tests’** directory, such as TestPC.py and TestGES.py.

For the implemented modules, such as (conditional) independent test methods, we provide unit tests for the convenience of developing your own methods.

# Benchmarks

For the convenience of our community, [CMU-CLeaR](https://www.cmu.edu/dietrich/causality) group maintains a list of benchmark datasets including real-world scenarios and various learning tasks. Please refer to the following links:

* [https://github.com/cmu-phil/example-causal-datasets](https://github.com/cmu-phil/example-causal-datasets) (maintained by Joseph Ramsey)
* [https://www.cmu.edu/dietrich/causality/causal-learn](https://www.cmu.edu/dietrich/causality/causal-learn>)

Please feel free to let us know if you have any recommendation regarding causal datasets with high-quality. We are grateful for any effort that benefits the development of causality community.


# Contribution

Please feel free to open an issue if you find anything unexpected.
And please create pull requests, perhaps after passing unittests in 'tests/', if you would like to contribute to causal-learn.
We are always targeting to make our community better!