from causallearn.graph.Graph import Graph


class SHD:
    """
    Compute the Structural Hamming Distance (SHD) between two graphs. In simple terms, this is the number of edge
    insertions, deletions or flips in order to transform one graph to another graph.
    """
    def __init__(self, truth: Graph, est: Graph):
        """
        Compute and store the Structural Hamming Distance (SHD) between two graphs.

        Parameters
        ----------
        truth : Graph
            Truth graph.
        est :
            Estimated graph.
        """
        truth_node_map = {node.get_name(): node_id for node, node_id in truth.node_map.items()}
        est_node_map = {node.get_name(): node_id for node, node_id in est.node_map.items()}
        assert set(truth_node_map.keys()) == set(est_node_map.keys()), "The two graphs have different sets of node names."

        self.__SHD: int = 0
        for node_i_name, truth_node_i_id in truth_node_map.items():
            for node_j_name, truth_node_j_id in truth_node_map.items():
                if truth_node_j_id < truth_node_i_id: continue  # we allow `==' to care about the possibly self-loops.
                est_node_i_id, est_node_j_id = est_node_map[node_i_name], est_node_map[node_j_name]
                truth_ij_edge_endpoints = (truth.graph[truth_node_i_id, truth_node_j_id], truth.graph[truth_node_j_id, truth_node_i_id])
                est_ij_edge_endpoints = (est.graph[est_node_i_id, est_node_j_id], est.graph[est_node_j_id, est_node_i_id])
                if truth_ij_edge_endpoints != est_ij_edge_endpoints: self.__SHD += 1

    def get_shd(self) -> int:
        return self.__SHD
