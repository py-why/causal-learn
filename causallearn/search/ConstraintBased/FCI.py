import warnings
from queue import Queue
from causallearn.graph.Edge import Edge
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *
from causallearn.utils import Fas
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
import numpy as np


class SepsetsPossibleDsep():
    def __init__(self, data, graph, independence_test, alpha, knowledge, depth, maxPathLength, verbose):
        self.data = data
        self.graph = graph
        self.independence_test = independence_test
        self.alpha = alpha
        self.knowledge = knowledge
        self.depth = depth
        self.maxPathLength = maxPathLength
        self.verbose = verbose


    def traverseSemiDirected(self, node, edge):
        if node == edge.get_node1():
            if edge.get_endpoint1() == Endpoint.TAIL or edge.get_endpoint1() == Endpoint.CIRCLE:
                return edge.get_node2()
        elif node == edge.get_node2():
            if edge.get_endpoint2() == Endpoint.TAIL or edge.get_endpoint2() == Endpoint.CIRCLE:
                return edge.get_node1()
        return None

    def existsSemidirectedPath(self, node_from, node_to, G):
        Q = Queue()
        V = set()
        Q.put(node_from)
        V.add(node_from)

        while not Q.empty():
            node_t = Q.get_nowait()
            if node_t == node_to:
                return True

            for node_u in G.get_adjacent_nodes(node_t):
                edge = G.get_edge(node_t, node_u)
                node_c = self.traverseSemiDirected(node_t, edge)

                if node_c is None:
                    continue

                if V.__contains__(node_c):
                    continue

                V.add(node_c)
                Q.put(node_c)

        return False

    def existOnePathWithPossibleParents(self, previous, node_w, node_x, node_b, graph):
        if node_w == node_x:
            return True
        p = previous.get(node_w)
        if p is None:
            return False
        for node_r in p:
            if node_r == node_b or node_r == node_x:
                continue

            if self.existsSemidirectedPath(node_r, node_x, graph) or self.existsSemidirectedPath(node_r, node_b, graph):
                if self.existOnePathWithPossibleParents(previous, node_r, node_x, node_b, graph):
                    return True
        return False


    def getPossibleDsep(self, node_x, node_y, maxPathLength):
        dsep = set()
        Q = Queue()
        V = set()
        previous = {node_x: None}
        e = None
        distance = 0

        for node_b in self.graph.get_adjacent_nodes(node_x):
            if node_b == node_y:
                continue
            edge = (node_x, node_b)
            if e is None:
                e = edge
            Q.put(edge)
            V.add(edge)

            # addToList
            node_list = previous.get(node_x)
            if node_list is None:
                node_list = []
            node_list.append(node_b)
            # previous[node_x] = node_list
            dsep.add(node_b)

        while not Q.empty():
            t = Q.get_nowait()
            if e == t:
                e = None
                distance += 1
                if distance > 0 and distance > (1000 if maxPathLength == -1 else maxPathLength):
                    break
            node_a, node_b = t

            if self.existOnePathWithPossibleParents(previous, node_b, node_x, node_b, self.graph):
                dsep.add(node_b)

            for node_c in self.graph.get_adjacent_nodes(node_b):
                if node_c == node_a:
                    continue
                if node_c == node_x:
                    continue
                if node_c == node_y:
                    continue

                # addToList
                node_list = previous.get(node_c)
                if node_list is None:
                    node_list = []
                node_list.append(node_b)
                # previous[node_c] = node_list

                # isDefCollider
                edge1 = self.graph.get_edge(node_a, node_b)
                edge2 = self.graph.get_edge(node_b, node_c)
                isDefCollider =  not (edge1 is None or edge2 is None) and \
                        edge1.get_proximal_endpoint(node_b) == Endpoint.ARROW and \
                        edge2.get_proximal_endpoint(node_b) == Endpoint.ARROW

                if isDefCollider or self.graph.is_adjacent_to(node_a, node_c):
                    u = (node_a, node_c)
                    if V.__contains__(u):
                        continue

                    V.add(u)
                    Q.put(u)

                    if e is None:
                        e = u

        if dsep.__contains__(node_x):
            dsep.remove(node_x)
        if dsep.__contains__(node_y):
            dsep.remove(node_y)

        if self.verbose:
            message = "Possible-D-Sep(" + node_x.get_name() + ", " + node_y.get_name() + ") = [ "
            for dsep_node in dsep:
                message += dsep_node.get_name() + " "
            message += "]"
            print(message)

        return dsep


    def possibleParentOf(self, node_z, node_x, bk):
        return True if bk is None else not(bk.is_forbidden(node_z, node_x) or bk.is_required(node_x, node_z))


    def possibleParents(self, node_x, nodes, knowledge):
        possibleParents = list()
        for node_z in nodes:
            if self.possibleParentOf(node_z, node_x, knowledge):
                possibleParents.append(node_z)
        return possibleParents


    def get_cond_set(self, node_1, node_2, max_path_length):
        possibleDsepSet = self.getPossibleDsep(node_1, node_2, max_path_length)
        possibleDsep = list(possibleDsepSet)
        noEdgeRequired = True if self.knowledge is None else not \
            (self.knowledge.is_required(node_1, node_2) or self.knowledge.is_required(node_2, node_1))

        possParents = self.possibleParents(node_1, possibleDsep, self.knowledge)

        _depth = 1000 if self.depth == -1 else self.depth

        possible_sep_set = set()

        for d in range(1 + min(_depth, len(possParents))):
            cg = ChoiceGenerator(len(possParents), d)
            choice = cg.next()
            flag = False
            while choice is not None:
                condSet = [self.graph.node_map[possParents[index]] for index in choice]
                choice = cg.next()

                X, Y = self.graph.node_map[node_1], self.graph.node_map[node_2]
                X, Y = (X, Y) if (X < Y) else (Y, X)
                XYS_key = (X, Y, frozenset(condSet))
                if XYS_key in Fas.citest_cache:
                    p_value = Fas.citest_cache[XYS_key]
                else:
                    p_value = self.independence_test(self.data, X, Y, tuple(condSet), Fas.cardinalities) if Fas.is_discrete \
                            else self.independence_test(self.data, X, Y, tuple(condSet))
                    Fas.citest_cache[XYS_key] = p_value
                independent = p_value > self.alpha

                if independent and noEdgeRequired:
                    for item in condSet:
                        possible_sep_set.add(item)
                    flag = True
            if flag:
                return possible_sep_set
        return None


    def get_sep_set(self, node_i, node_k):
        condSet = self.get_cond_set(node_i, node_k, self.maxPathLength)
        if condSet is None:
            condSet = self.get_cond_set(node_k, node_i, self.maxPathLength)
        return condSet


def fci_orient_bk(bk, graph):
    if bk is None:
        return
    print("Starting BK Orientation.")
    edges = graph.get_graph_edges()
    for edge in edges:
        if bk.is_forbidden(edge.get_node1(), edge.get_node2()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node2(), edge.get_node1())
            print("Orienting edge (Knowledge): " + graph.get_edge(edge.get_node2(), edge.get_node1()))
        elif bk.is_forbidden(edge.get_node2(), edge.get_node1()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node1(), edge.get_node2())
            print("Orienting edge (Knowledge): " + graph.get_edge(edge.get_node2(), edge.get_node1()))
        elif bk.is_required(edge.get_node1(), edge.get_node2()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node1(), edge.get_node2())
            print("Orienting edge (Knowledge): " + graph.get_edge(edge.get_node2(), edge.get_node1()))
        elif bk.is_required(edge.get_node2(), edge.get_node1()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node2(), edge.get_node1())
            print("Orienting edge (Knowledge): " + graph.get_edge(edge.get_node2(), edge.get_node1()))
    print("Finishing BK Orientation.")


def is_arrow_point_allowed(node_x, node_y, graph, knowledge):
    if graph.get_endpoint(node_x, node_y) == Endpoint.ARROW:
        return True
    if graph.get_endpoint(node_x, node_y) == Endpoint.TAIL:
        return False
    if graph.get_endpoint(node_y, node_x) == Endpoint.ARROW:
        if knowledge is not None and not knowledge.is_forbidden(node_x, node_y):
            return True
    if graph.get_endpoint(node_y, node_x) == Endpoint.TAIL:
        if knowledge is not None and not knowledge.is_forbidden(node_x, node_y):
            return True
    return graph.get_endpoint(node_x, node_y) == Endpoint.CIRCLE

def rule0(graph, nodes, sep_sets, knowledge, verbose):
    reorientAllWith(graph, Endpoint.CIRCLE)
    fci_orient_bk(knowledge, graph)
    for node_b in nodes:
        adjacent_nodes = graph.get_adjacent_nodes(node_b)
        if len(adjacent_nodes) < 2:
            continue

        cg = ChoiceGenerator(len(adjacent_nodes), 2)
        combination = cg.next()
        while combination is not None:
            node_a = adjacent_nodes[combination[0]]
            node_c = adjacent_nodes[combination[1]]
            combination = cg.next()

            if graph.is_adjacent_to(node_a, node_c):
                continue
            if graph.is_def_collider(node_a, node_b, node_c):
                continue
            # check if is collider
            sep_set = sep_sets.get((graph.node_map[node_a], graph.node_map[node_c]))
            if sep_set is not None and not sep_set.__contains__(graph.node_map[node_b]):
                if not is_arrow_point_allowed(node_a, node_b, graph, knowledge):
                    continue
                if not is_arrow_point_allowed(node_c, node_b, graph, knowledge):
                    continue

                edge1 = graph.get_edge(node_a, node_b)
                graph.remove_edge(edge1)
                graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

                edge2 = graph.get_edge(node_c, node_b)
                graph.remove_edge(edge2)
                graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

                if verbose:
                    print(
                        "Orienting collider: " + node_a.get_name() + " *-> " + node_b.get_name() + " <-* " + node_c.get_name())

def reorientAllWith(graph, endpoint):
    # reorient all edges with CIRCLE Endpoint
    ori_edges = graph.get_graph_edges()
    for ori_edge in ori_edges:
        graph.remove_edge(ori_edge)
        ori_edge.set_endpoint1(endpoint)
        ori_edge.set_endpoint2(endpoint)
        graph.add_edge(ori_edge)


def ruleR1(node_a, node_b, node_c, graph, bk, changeFlag, verbose=False):
    if graph.is_adjacent_to(node_a, node_c):
        return changeFlag

    if graph.get_endpoint(node_a, node_b) == Endpoint.ARROW and graph.get_endpoint(node_c, node_b) == Endpoint.CIRCLE:
        if not is_arrow_point_allowed(node_b, node_c, graph, bk):
            return changeFlag

        edge1 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_c, node_b, Endpoint.ARROW, Endpoint.TAIL))

        changeFlag = True

        if verbose:
            print("Orienting edge (Away from collider):" + graph.get_edge(node_b, node_c).__str__())

    return changeFlag


def ruleR2(node_a, node_b, node_c, graph, bk, changeFlag, verbose=False):
    if graph.is_adjacent_to(node_a, node_c) and graph.get_endpoint(node_a, node_c) == Endpoint.CIRCLE:
        if graph.get_endpoint(node_a, node_b) == Endpoint.ARROW and \
                graph.get_endpoint(node_b, node_c) == Endpoint.ARROW and \
                (graph.get_endpoint(node_b, node_a) == Endpoint.TAIL or
                 graph.get_endpoint(node_c, node_b) == Endpoint.TAIL):
            if not is_arrow_point_allowed(node_a, node_c, graph, bk):
                return changeFlag

            edge1 = graph.get_edge(node_a, node_c)
            graph.remove_edge(edge1)
            graph.add_edge(Edge(node_a, node_c, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

            if verbose:
                print("Orienting edge (Away from ancestor): " + graph.get_edge(node_a, node_c).__str__())

            changeFlag = True

    return changeFlag


def rulesR1R2cycle(graph, bk, changeFlag, verbose=False):
    nodes = graph.get_nodes()
    for node_B in nodes:
        adj = graph.get_adjacent_nodes(node_B)

        if len(adj) < 2:
            continue

        cg = ChoiceGenerator(len(adj), 2)
        combination = cg.next()

        while combination is not None:
            node_A = adj[combination[0]]
            node_C = adj[combination[1]]
            combination = cg.next()

            changeFlag = ruleR1(node_A, node_B, node_C, graph, bk, changeFlag, verbose)
            changeFlag = ruleR1(node_C, node_B, node_A, graph, bk, changeFlag, verbose)
            changeFlag = ruleR2(node_A, node_B, node_C, graph, bk, changeFlag, verbose)
            changeFlag = ruleR2(node_C, node_B, node_A, graph, bk, changeFlag, verbose)

    return changeFlag


def isNoncollider(graph, sep_sets, node_i, node_j, node_k):
    sep_set = sep_sets[(graph.node_map[node_i], graph.node_map[node_k])]
    return sep_set is not None and sep_set.__contains__(graph.node_map[node_j])


def ruleR3(graph, sep_sets, bk, changeFlag, verbose=False):
    nodes = graph.get_nodes()
    for node_B in nodes:
        intoBArrows = graph.get_nodes_into(node_B, Endpoint.ARROW)
        intoBCircles = graph.get_nodes_into(node_B, Endpoint.CIRCLE)

        for node_D in intoBCircles:
            if len(intoBArrows) < 2:
                continue
            gen = ChoiceGenerator(len(intoBArrows), 2)
            choice = gen.next()

            while choice is not None:
                node_A = intoBArrows[choice[0]]
                node_C = intoBArrows[choice[1]]
                choice = gen.next()

                if graph.is_adjacent_to(node_A, node_C):
                    continue

                if not graph.is_adjacent_to(node_A, node_D) or graph.is_adjacent_to(node_C, node_D):
                    continue

                if not isNoncollider(graph, sep_sets, node_A, node_D, node_C):
                    continue

                if graph.get_endpoint(node_A, node_D) != Endpoint.CIRCLE:
                    continue

                if graph.get_endpoint(node_C, node_D) != Endpoint.CIRCLE:
                    continue

                if not is_arrow_point_allowed(node_D, node_B, graph, bk):
                    continue

                edge1 = graph.get_edge(node_D, node_B)
                graph.remove_edge(edge1)
                graph.add_edge(Edge(node_D, node_B, edge1.get_proximal_endpoint(node_D), Endpoint.ARROW))

                if verbose:
                    print("Orienting edge (Double triangle): " + graph.get_edge(node_D, node_B).__str__())

                changeFlag = True
    return changeFlag


def getPath(node_c, previous):
    l = []
    node_p = previous[node_c]
    if node_p is not None:
        l.append(node_p)
    while node_p is not None:
        node_p = previous.get(node_p)
        if node_p is not None:
            l.append(node_p)
    return l



def doDdpOrientation(node_d, node_a, node_b, node_c, previous, graph, data, independence_test_method, alpha, sep_sets, changeFlag, bk, verbose=False):
    if graph.is_adjacent_to(node_d, node_c):
        raise Exception("illegal argument!")
    path = getPath(node_d, previous)

    X, Y = graph.node_map[node_d], graph.node_map[node_c]
    X, Y = (X, Y) if (X < Y) else (Y, X)
    condSet = tuple([graph.node_map[nn] for nn in path])
    XYS_key = (X, Y, frozenset(condSet))
    if XYS_key in Fas.citest_cache:
        p_value = Fas.citest_cache[XYS_key]
    else:
        p_value = independence_test_method(data, X, Y, condSet, Fas.cardinalities) if Fas.is_discrete \
                    else independence_test_method(data, X, Y, condSet)
        Fas.citest_cache[XYS_key] = p_value
    ind = p_value > alpha

    path2 = list(path)
    path2.remove(node_b)

    X, Y = graph.node_map[node_d], graph.node_map[node_c]
    X, Y = (X, Y) if (X < Y) else (Y, X)
    condSet = tuple([graph.node_map[nn2] for nn2 in path2])
    XYS_key = (X, Y, frozenset(condSet))
    if XYS_key in Fas.citest_cache:
        p_value2 = Fas.citest_cache[XYS_key]
    else:
        p_value2 = independence_test_method(data, X, Y, condSet, Fas.cardinalities) if Fas.is_discrete \
                    else independence_test_method(data, X, Y, condSet)
        Fas.citest_cache[XYS_key] = p_value2
    ind2 = p_value2 > alpha

    if not ind and not ind2:
        sep_set = sep_sets.get((graph.node_map[node_d], graph.node_map[node_c]))
        if verbose:
            message = "Sepset for d = " + node_d.get_name() + " and c = " + node_c.get_name() + " = [ "
            if sep_set is not None:
                for ss in sep_set:
                    message += graph.nodes[ss].get_name() + " "
            message += "]"
            print(message)

        if sep_set is None:
            if verbose:
                print("Must be a sepset: " + node_d.get_name() + " and " + node_c.get_name() + "; they're non-adjacent.")
            return False, changeFlag

        ind = sep_set.__contains__(graph.node_map[node_b])

    if ind:
        edge = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge)
        graph.add_edge(Edge(node_c, node_b, edge.get_proximal_endpoint(node_c), Endpoint.TAIL))

        if verbose:
            print("Orienting edge (Definite discriminating path d = " + node_d.get_name() + "): " + graph.get_edge(node_b, node_c).__str__())

        changeFlag = True
        return True, changeFlag
    else:
        if not is_arrow_point_allowed(node_a, node_b, graph, bk):
            return False, changeFlag

        if not is_arrow_point_allowed(node_c, node_b, graph, bk):
            return False, changeFlag

        edge1 = graph.get_edge(node_a, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

        edge2 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge2)
        graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

        if verbose:
            print("Orienting collider (Definite discriminating path.. d = " + node_d.get_name() + "): " + node_a.get_name() + " *-> " + node_b.get_name() + " <-* " + node_c.get_name())

        changeFlag = True
        return True, changeFlag


def ddpOrient(node_a, node_b, node_c, graph, maxPathLength, data, independence_test_method, alpha, sep_sets, changeFlag, bk, verbose=False):
    Q = Queue()
    V = set()
    e = None
    distance = 0
    previous = {}

    cParents = graph.get_parents(node_c)

    Q.put(node_a)
    V.add(node_a)
    V.add(node_b)
    previous[node_a] = node_b

    while not Q.empty():
        node_t = Q.get_nowait()

        if e is None or e == node_t:
            e = node_t
            distance += 1
            if distance > 0 and distance > (1000 if maxPathLength == -1 else maxPathLength):
                return changeFlag

        nodesInTo = graph.get_nodes_into(node_t, Endpoint.ARROW)

        for node_d in nodesInTo:
            if V.__contains__(node_d):
                continue

            previous[node_d] = node_t
            node_p = previous[node_t]

            if not graph.is_def_collider(node_d, node_t, node_p):
                continue

            previous[node_d] = node_t

            if not graph.is_adjacent_to(node_d, node_c) and node_d != node_c:
                res, changeFlag = doDdpOrientation(node_d, node_a, node_b, node_c, previous, graph, data, independence_test_method, alpha, sep_sets, changeFlag, bk, verbose)
                if res:
                    return changeFlag

            if cParents.__contains__(node_d):
                Q.put(node_d)
                V.add(node_d)
    return changeFlag


def ruleR4B(graph, maxPathLength, data, independence_test_method, alpha, sep_sets, changeFlag, bk, verbose=False):
    nodes = graph.get_nodes()

    for node_b in nodes:
        possA = graph.get_nodes_out_of(node_b, Endpoint.ARROW)
        possC = graph.get_nodes_into(node_b, Endpoint.CIRCLE)

        for node_a in possA:
            for node_c in possC:
                if not graph.is_parent_of(node_a, node_c):
                    continue

                if graph.get_endpoint(node_b, node_c) != Endpoint.ARROW:
                    continue

                changeFlag = ddpOrient(node_a, node_b, node_c, graph, maxPathLength, data, independence_test_method, alpha, sep_sets, changeFlag, bk, verbose)
    return changeFlag


def fci(dataset, independence_test_method = fisherz, alpha=0.05, depth=-1, max_path_length=-1,
        verbose=False, background_knowledge=None):
    '''
    Causal Discovery with Fast Causal Inference

    Parameters
    ----------
    dataset: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    independence_test_method: the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: Significance level of independence tests(p_value)([0,1])
    depth: The depth for the fast adjacency search, or -1 if unlimited
    max_path_length: the maximum length of any discriminating path, or -1 if unlimited.
    verbose: True is verbose output should be printed or logged
    background_knowledge: background knowledge

    Returns
    -------
    graph : a CausalGraph object, where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j,
                    cg.G.graph[j,i]=2 and cg.G.graph[i,j]=1 indicates  i o-> j.
    '''

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    Fas.citest_cache = dict()  # DEBUG@2021/12/23, must refresh cache every time at initialization
    Fas.cardinalities = None
    Fas.is_discrete = False

    def _unique(column):
        return np.unique(column, return_inverse=True)[1]

    if independence_test_method == chisq or independence_test_method == gsq:
        dataset = np.apply_along_axis(_unique, 0, dataset).astype(np.int64)
        Fas.is_discrete = True
        Fas.cardinalities = np.max(dataset, axis=0) + 1


    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    if (max_path_length is not None) and type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------

    nodes = []
    for i in range(dataset.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)

    # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
    graph, sep_sets = Fas.fas(dataset, nodes, independence_test_method=independence_test_method, alpha=alpha, knowledge=background_knowledge, depth=depth, verbose=verbose)

    # reorient all edges with CIRCLE Endpoint
    ori_edges = graph.get_graph_edges()
    for ori_edge in ori_edges:
        graph.remove_edge(ori_edge)
        ori_edge.set_endpoint1(Endpoint.CIRCLE)
        ori_edge.set_endpoint2(Endpoint.CIRCLE)
        graph.add_edge(ori_edge)

    sp = SepsetsPossibleDsep(dataset, graph, independence_test_method, alpha, background_knowledge, depth, max_path_length, verbose)

    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    waiting_to_deleted_edges = []

    for edge in graph.get_graph_edges():
        node_x = edge.get_node1()
        node_y = edge.get_node2()

        sep_set = sp.get_sep_set(node_x, node_y)

        if sep_set is not None:
            waiting_to_deleted_edges.append((node_x, node_y, sep_set))

    for waiting_to_deleted_edge in waiting_to_deleted_edges:
        dedge_node_x, dedge_node_y, dedge_sep_set = waiting_to_deleted_edge
        graph.remove_edge(graph.get_edge(dedge_node_x, dedge_node_y))
        sep_sets[(graph.node_map[dedge_node_x], graph.node_map[dedge_node_y])] = dedge_sep_set

        if verbose:
            message = "Possible DSEP Removed " + dedge_node_x.get_name() + " --- " + dedge_node_y.get_name() + " sepset = ["
            for ss in dedge_sep_set:
                message += graph.nodes[ss].get_name() + " "
            message += "]"
            print(message)

    reorientAllWith(graph, Endpoint.CIRCLE)
    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    changeFlag = True
    firstTime = True

    while changeFlag:
        changeFlag = False
        changeFlag = rulesR1R2cycle(graph, background_knowledge, changeFlag, verbose)
        changeFlag = ruleR3(graph, sep_sets, background_knowledge, changeFlag, verbose)

        if changeFlag or (firstTime and background_knowledge is not None and
                          len(background_knowledge.forbidden_rules_specs) > 0 and
                          len(background_knowledge.required_rules_specs) > 0 and
                          len(background_knowledge.tier_map.keys()) > 0):
            changeFlag = ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets, changeFlag, background_knowledge, verbose)

            firstTime = False

            if verbose:
                print("Epoch")

    graph.set_pag(True)

    return graph

