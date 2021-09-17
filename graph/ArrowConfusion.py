from graph.Graph import Graph
from graph.Endpoint import Endpoint

import numpy as np

class ArrowConfusion:
    __arrowsFp = 0
    __arrowsFn = 0
    __arrowsTp = 0
    __arrowsTn = 0

    __arrowsFpCE = 0
    __arrowsFnCE = 0
    __arrowsTpCE = 0
    __arrowsTnCE = 0

    def __init__(self, truth:Graph, est:Graph):
        nodes = truth.get_nodes()

        truePositives = np.zeros(len(nodes), len(nodes))
        estPositives = np.zeros(len(nodes), len(nodes))
        truePositivesCE = np.zeros(len(nodes), len(nodes))
        estPositivesCE = np.zeros(len(nodes), len(nodes))

        # Assumes the the list of nodes for the two graphs are the same.
        for i in list(range(1, len(nodes))):
            for j in list(range(1, len(nodes))):
                if truth.get_endpoint(nodes[i], nodes[j]) == Endpoint.ARROW:
                    truePositives[i][j] = 1
                if est.get_endpoint(nodes[i], nodes[j]) == Endpoint.ARROW:
                    estPositives[i][j] = 1
                if truth.get_endpoint(nodes[i], nodes[j]) == Endpoint.ARROW \
                        and est.is_adjacent_to(nodes[i], nodes[j]):
                    truePositivesCE[i][j] = 1
                if est.get_endpoint(nodes[i], nodes[j]) == Endpoint.ARROW \
                        and truth.is_adjacent_to(nodes[i], nodes[j]):
                    estPositives[i][j] = 1

        self.__arrowsFp = (estPositives - truePositives).sum()
        self.__arrowsFn = (truePositives - estPositives).sum()
        self.__arrowsTp = truePositives.sum()
        self.__arrowsTn = (np.ones(len(nodes), len(nodes)) - truePositives).sum()

        self.__arrowsFpCE = (estPositivesCE - truePositivesCE).sum()
        self.__arrowsFnCE = (truePositivesCE - estPositivesCE).sum()
        self.__arrowsTpCE = truePositivesCE.sum()
        self.__arrowsTnCE = (np.ones(len(nodes), len(nodes)) - truePositivesCE).sum()

    def getArrowsFp(self):
        return self.__arrowsFp

    def getArrowsFn(self):
        return self.__arrowsFn

    def getArrowsTp(self):
        return self.__arrowsTp

    def getArrowsTn(self):
        return self.__arrowsTn

    def getArrowsFpCE(self):
        return self.__arrowsFpCE

    def getArrowsFnCE(self):
        return self.__arrowsFnCE

    def getArrowsTpCE(self):
        return self.__arrowsTpCE

    def getArrowsTnCE(self):
        return self.__arrowsTnCE


