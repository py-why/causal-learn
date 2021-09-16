from graph.Graph import Graph

class AdjacencyConfusion:
    __adjFn = 0
    __adjTp = 0
    __adjFp = 0
    __adjTn = 0

    def __init__(self, truth:Graph, est:Graph):
        nodes = truth.get_nodes()
        for i in list(range(1, len(nodes))):
            for j in list(range(i+1,len(nodes))):
                if not est.is_adjacent_to(nodes[i], nodes[j]):
                    self.__adjFn = self.__adjFn + 1
                else:
                    self.__adjTp = self.__adjTp + 1
                if not truth.is_adjacent_to(nodes[i], nodes[j]):
                    self.__adjFp = self.__adjFp + 1
        allEdges = truth.get_num_nodes() * (truth.get_num_nodes() - 1) / 2
        self.__adjTn = allEdges - self.__adjFn

    def getAdjTp(self):
        return self.__adjTp

    def getAdjFp(self):
        return self.__adjFp

    def getAdjFn(self):
        return self.__adjFn

    def getAdjTn(self):
        return self.__adjTn
