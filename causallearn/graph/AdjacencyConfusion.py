from causallearn.graph.Graph import Graph


class AdjacencyConfusion:
    '''
    Compute the adjacency confusion between two graphs.
    '''
    __adjFn = 0
    __adjTp = 0
    __adjFp = 0
    __adjTn = 0

    def __init__(self, truth: Graph, est: Graph):
        '''
        Compute and store the adjacency confusion between two graphs.

        Parameters
        ----------
        truth : Graph
            Truth graph.
        est :
            Estimated graph.
        '''
        nodes = truth.get_nodes()
        nodes_name = [node.get_name() for node in nodes]
        for i in list(range(0, len(nodes))):
            for j in list(range(i + 1, len(nodes))):
                estAdj = est.is_adjacent_to(est.get_node(nodes_name[i]), est.get_node(nodes_name[j]))
                truthAdj = truth.is_adjacent_to(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j]))

                if truthAdj and not estAdj:
                    self.__adjFn = self.__adjFn + 1
                elif estAdj and not truthAdj:
                    self.__adjFp = self.__adjFp + 1
                elif estAdj and truthAdj:
                    self.__adjTp = self.__adjTp + 1
                elif not estAdj and not truthAdj:
                    self.__adjTn = self.__adjTn + 1

    def get_adj_tp(self):
        return self.__adjTp

    def get_adj_fp(self):
        return self.__adjFp

    def get_adj_fn(self):
        return self.__adjFn

    def get_adj_tn(self):
        return self.__adjTn

    def get_adj_precision(self):
        return self.__adjTp / (self.__adjTp + self.__adjFp)

    def get_adj_recall(self):
        return self.__adjTp / (self.__adjTp + self.__adjFn)
