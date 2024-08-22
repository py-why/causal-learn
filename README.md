# causal-learn: Causal Discovery in Python

Causal-learn ([documentation](https://causal-learn.readthedocs.io/en/latest/), [paper](https://jmlr.org/papers/volume25/23-0970/23-0970.pdf)) is a python package for causal discovery that implements both classical and state-of-the-art causal discovery algorithms, which is a Python translation and extension of [Tetrad](https://github.com/cmu-phil/tetrad).

The package is actively being developed. Feedbacks (issues, suggestions, etc.) are highly encouraged.

# Package Overview

Our causal-learn implements methods for causal discovery:

* Constraint-based causal discovery methods.
* Score-based causal discovery methods.
* Causal discovery methods based on constrained functional causal models.
* Hidden causal representation learning.
* Permutation-based causal discovery methods.
* Granger causality.
* Multiple utilities for building your own method, such as independence tests, score functions, graph operations, and evaluations.

# Install

Causal-learn needs the following packages to be installed beforehand:

* python 3 (>=3.7)
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

To use causal-learn, we could install it using [pip](https://pypi.org/project/causal-learn/):

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
* [https://www.cmu.edu/dietrich/causality/projects/causal_learn_benchmarks](https://www.cmu.edu/dietrich/causality/projects/causal_learn_benchmarks)

Please feel free to let us know if you have any recommendation regarding causal datasets with high-quality. We are grateful for any effort that benefits the development of causality community.


# Contribution

Please feel free to open an issue if you find anything unexpected.
And please create pull requests, perhaps after passing unittests in 'tests/', if you would like to contribute to causal-learn.
We are always targeting to make our community better!

# Running Tetrad in Python

Although causal-learn provides python implementations for some causal discovery algorithms, there are currently a lot more in the classical Java-based [Tetrad](https://github.com/cmu-phil/tetrad) program. For users who would like to incorporate arbitrary Java code in Tetrad as part of a Python workflow, we strongly recommend considering [py-tetrad](https://github.com/cmu-phil/py-tetrad). Here is a list of [reusable examples](https://github.com/cmu-phil/py-tetrad/tree/main/pytetrad) of how to painlessly benefit from the most comprehensive Tetrad Java codebase.

# Citation

Please cite as:

```
@article{zheng2024causal,
  title={Causal-learn: Causal discovery in python},
  author={Zheng, Yujia and Huang, Biwei and Chen, Wei and Ramsey, Joseph and Gong, Mingming and Cai, Ruichu and Shimizu, Shohei and Spirtes, Peter and Zhang, Kun},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={60},
  pages={1--8},
  year={2024}
}
```