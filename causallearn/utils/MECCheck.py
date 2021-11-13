from copy import deepcopy

import numpy as np

from causallearn.utils.DAG2CPDAG import dag2cpdag


def mec_check(DAG1, DAG2):
    '''
    Check whether DAG1 and DAG2 are belong to the same Markov Equivalence Class

    Parameters
    ----------
    DAG1, DAG2: Direct Acyclic Graph

    Returns
    -------
    True when DAG1 and DAG2 are belong to the same Markov Equivalence Class
    else False
    '''

    G1 = deepcopy(DAG1)
    G2 = deepcopy(DAG2)
    CPDAG1 = dag2cpdag(G1)
    CPDAG2 = dag2cpdag(G2)
    if np.all(CPDAG1.graph == CPDAG2.graph):
        return True
    else:
        return False
