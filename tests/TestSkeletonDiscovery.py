from unittest import TestCase
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
import networkx as nx
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation


class TestSkeletonDiscovery(TestCase):
    def test_sepset(self):
        truth_DAG_directed_edges = {(0, 2), (1, 2), (2, 3), (2, 4)}

        true_dag_netx = nx.DiGraph()
        true_dag_netx.add_nodes_from(list(range(5)))
        true_dag_netx.add_edges_from(truth_DAG_directed_edges)

        data = np.zeros((100, len(true_dag_netx.nodes)))  # just a placeholder
        cg = pc(data, 0.05, d_separation, True, 0, -1, true_dag=true_dag_netx)
        assert cg.sepset[0, 2] is None