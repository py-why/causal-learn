from copy import deepcopy

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint


def meek(cg, background_knowledge=None):
    '''
    Run Meek rules

    Parameters
    ----------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    '''

    cg_new = deepcopy(cg)

    UT = cg_new.find_unshielded_triples()
    Tri = cg_new.find_triangles()
    Kite = cg_new.find_kites()

    Loop = True

    while Loop:
        Loop = False
        for (i, j, k) in UT:
            if cg_new.is_fully_directed(i, j) and cg_new.is_undirected(j, k):
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[j], cg_new.G.nodes[k]) or
                         background_knowledge.is_required(cg_new.G.nodes[k], cg_new.G.nodes[j])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[k])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[j], cg_new.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    Loop = True

        for (i, j, k) in Tri:
            if cg_new.is_fully_directed(i, j) and cg_new.is_fully_directed(j, k) and cg_new.is_undirected(i, k):
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[k]) or
                         background_knowledge.is_required(cg_new.G.nodes[k], cg_new.G.nodes[i])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[k])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    Loop = True

        for (i, j, k, l) in Kite:
            if cg_new.is_undirected(i, j) and cg_new.is_undirected(i, k) and cg_new.is_fully_directed(j, l) \
                    and cg_new.is_fully_directed(k, l) and cg_new.is_undirected(i, l):
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[l]) or
                         background_knowledge.is_required(cg_new.G.nodes[l], cg_new.G.nodes[i])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[l])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[l], Endpoint.TAIL, Endpoint.ARROW))
                    Loop = True

    return cg_new


def definite_meek(cg, background_knowledge=None):
    '''
    Run Meek rules over the definite unshielded triples

    Parameters
    ----------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    '''

    cg_new = deepcopy(cg)

    Tri = cg_new.find_triangles()
    Kite = cg_new.find_kites()

    Loop = True

    while Loop:
        Loop = False
        for (i, j, k) in cg_new.definite_non_UC:
            if cg_new.is_fully_directed(i, j) and \
                    cg_new.is_undirected(j, k) and \
                    not ((background_knowledge is not None) and
                         (background_knowledge.is_forbidden(cg_new.G.nodes[j], cg_new.G.nodes[k]) or
                          background_knowledge.is_required(cg_new.G.nodes[k], cg_new.G.nodes[j]))):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[k])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[j], cg_new.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                Loop = True
            elif cg_new.is_fully_directed(k, j) and \
                    cg_new.is_undirected(j, i) and \
                    not ((background_knowledge is not None) and
                         (background_knowledge.is_forbidden(cg_new.G.nodes[j], cg_new.G.nodes[i]) or
                          background_knowledge.is_required(cg_new.G.nodes[i], cg_new.G.nodes[j]))):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[i])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[j], cg_new.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW))
                Loop = True

        for (i, j, k) in Tri:
            if cg_new.is_fully_directed(i, j) and cg_new.is_fully_directed(j, k) and cg_new.is_undirected(i, k):
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[k]) or
                         background_knowledge.is_required(cg_new.G.nodes[k], cg_new.G.nodes[i])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[k])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    Loop = True

        for (i, j, k, l) in Kite:
            if ((j, l, k) in cg_new.definite_UC or (k, l, j) in cg_new.definite_UC) \
                    and ((j, i, k) in cg_new.definite_non_UC or (k, i, j) in cg_new.definite_non_UC) \
                    and cg_new.is_undirected(i, l):
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[l]) or
                         background_knowledge.is_required(cg_new.G.nodes[l], cg_new.G.nodes[i])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[l])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[l], Endpoint.TAIL, Endpoint.ARROW))
                    Loop = True

    return cg_new
