from copy import deepcopy

import numpy as np

from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2CPDAG import dag2cpdag


def mec_check(DAG1: Dag, DAG2: Dag) -> bool:
    """
    Check whether DAG1 and DAG2 are belong to the same Markov Equivalence Class

    Parameters
    ----------
    DAG1, DAG2: Direct Acyclic Graph

    Returns
    -------
    True when DAG1 and DAG2 belong to the same Markov Equivalence Class
    else False
    """

    g1 = deepcopy(DAG1)
    g2 = deepcopy(DAG2)
    cpdag_1 = dag2cpdag(g1)
    cpdag_2 = dag2cpdag(g2)
    if np.all(cpdag_1.graph == cpdag_2.graph):
        return True
    else:
        return False
