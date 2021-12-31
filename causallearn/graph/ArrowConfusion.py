import numpy as np

from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph


class ArrowConfusion:
    '''
    Compute the arrow confusion between two graphs.
    '''
    __arrowsFp = 0
    __arrowsFn = 0
    __arrowsTp = 0
    __arrowsTn = 0

    __arrowsFpCE = 0
    __arrowsFnCE = 0
    __arrowsTpCE = 0
    __arrowsTnCE = 0

    def __init__(self, truth: Graph, est: Graph):
        '''
        Compute and store the arrow confusion between two graphs.

        Parameters
        ----------
        truth : Graph
            Truth graph.
        est :
            Estimated graph.
        '''
        nodes = truth.get_nodes()
        nodes_name = [node.get_name() for node in nodes]

        truePositives = np.zeros((len(nodes), len(nodes)))
        estPositives = np.zeros((len(nodes), len(nodes)))
        truePositivesCE = np.zeros((len(nodes), len(nodes)))
        estPositivesCE = np.zeros((len(nodes), len(nodes)))

        # Assumes the the list of nodes for the two graphs are the same.
        for i in list(range(0, len(nodes))):
            for j in list(range(0, len(nodes))):
                if truth.get_endpoint(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j])) == Endpoint.ARROW:
                    truePositives[j][i] = 1
                if est.get_endpoint(est.get_node(nodes_name[i]), est.get_node(nodes_name[j])) == Endpoint.ARROW:
                    estPositives[j][i] = 1
                if truth.get_endpoint(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j])) == Endpoint.ARROW \
                        and est.is_adjacent_to(est.get_node(nodes_name[i]), est.get_node(nodes_name[j])):
                    truePositivesCE[j][i] = 1
                if est.get_endpoint(est.get_node(nodes_name[i]), est.get_node(nodes_name[j])) == Endpoint.ARROW \
                        and truth.is_adjacent_to(truth.get_node(nodes_name[i]), truth.get_node(nodes_name[j])):
                    estPositivesCE[j][i] = 1

        ones = np.ones((len(nodes), len(nodes)))
        zeros = np.zeros((len(nodes), len(nodes)))

        self.__arrowsFp = (np.maximum(estPositives - truePositives, zeros)).sum()
        self.__arrowsFn = (np.maximum(truePositives - estPositives, zeros)).sum()
        self.__arrowsTp = (np.minimum(truePositives == estPositives, truePositives)).sum()
        self.__arrowsTn = (truePositives == estPositives).sum() - self.__arrowsTp

        self.__arrowsFpCE = (np.maximum(estPositivesCE - truePositivesCE, zeros)).sum()
        self.__arrowsFnCE = (np.maximum(truePositivesCE - estPositivesCE, zeros)).sum()
        self.__arrowsTpCE = (np.minimum(truePositivesCE == estPositivesCE, truePositivesCE)).sum()
        self.__arrowsTnCE = (truePositivesCE == estPositivesCE).sum() - self.__arrowsTpCE

    def get_arrows_fp(self):
        return self.__arrowsFp

    def get_arrows_fn(self):
        return self.__arrowsFn

    def get_arrows_tp(self):
        return self.__arrowsTp

    def get_arrows_tn(self):
        return self.__arrowsTn

    def get_arrows_fp_ce(self):
        return self.__arrowsFpCE

    def get_arrows_fn_ce(self):
        return self.__arrowsFnCE

    def get_arrows_tp_ce(self):
        return self.__arrowsTpCE

    def get_arrows_tn_ce(self):
        return self.__arrowsTnCE

    def get_arrows_precision(self):
        return self.__arrowsTp / (self.__arrowsTp + self.__arrowsFp)

    def get_arrows_recall(self):
        return self.__arrowsTp / (self.__arrowsTp + self.__arrowsFn)

    def get_arrows_precision_ce(self):
        return self.__arrowsTpCE / (self.__arrowsTpCE + self.__arrowsFpCE)

    def get_arrows_recall_ce(self):
        return self.__arrowsTpCE / (self.__arrowsTpCE + self.__arrowsFnCE)
