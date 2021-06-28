#######################################################################################################################
from tetradpy.graph import GeneralGraph
from tetradpy.graph import GraphUtils
from tetradpy.graph import Edge
#######################################################################################################################


def Meek(cg):
    """ Run Meek rules
    :param cg: a CausalGraph object
    :return:
    cg_new: a CausalGraph object
    """

    cg_new = GeneralGraph()
    cg_new.transfer_nodes_and_edges(cg)

    utils = GraphUtils()

    UT = utils.find_unshielded_triples(cg_new)
    Tri = utils.find_triangles(cg_new)
    Kite = utils.find_kites(cg_new)

    Loop = True

    while Loop:
        Loop = False
        for (i, j, k) in UT:
            if cg_new.is_directed_from_to(i, j)and cg_new.is_undirected_from_to(j, k):
                cg_new.add_edge(Edge(j, k, -1, 1))
                Loop = True

        for (i, j, k) in Tri:
            if cg_new.is_directed_from_to(i, j) and cg_new.is_directed_from_to(j, k) and cg_new.is_undirected_from_to(i, k):
                cg_new.add_edge(Edge(i, k, -1, 1))
                Loop = True

        for (i, j, k, l) in Kite:
            if cg_new.is_directed_from_to(i, j) and cg_new.is_undirected_from_to(i, k) and cg_new.is_directed_from_to(j, l)\
                    and cg_new.is_directed_from_to(k, l) and cg_new.is_undirected_from_to(i, l):
                cg_new.add_edge(Edge(i, l, -1, 1))
                Loop = True

    return cg_new

#######################################################################################################################


def definite_Meek(cg):
    """ Run Meek rules over the definite unshielded triples
    :param cg: a CausalGraph object
    :return:
    cg_new: a CausalGraph object
    """
    cg_new = GeneralGraph()
    cg_new.transfer_nodes_and_edges(cg)

    utils = GraphUtils()

    Tri = utils.find_triangles(cg_new)
    Kite = utils.find_kites(cg_new)

    Loop = True

    while Loop:
        Loop = False
        for (i, j, k) in cg_new.definite_non_UC:
            if cg_new.is_directed_from_to(i, j) and cg_new.is_undirected_from_to(j, k):
                cg_new.add_edge(j, k, -1, 1)
                Loop = True
            elif cg_new.is_directed_from_to(k, j) and cg_new.is_undirected_from_to(j, i):
                cg_new.add_edge(j, i, -1, 1)
                Loop = True

        for (i, j, k) in Tri:
            if cg_new.is_directed_from_to(i, j) and cg_new.is_directed_from_to(j, k) and cg_new.is_undirected_from_to(i, k):
                cg_new.add_edge(i, k, -1, 1)
                Loop = True

        for (i, j, k, l) in Kite:
            if (cg_new.is_definite_unshielded_collider(j, l, k) or cg_new.is_definite_unshielded_collider(k, l, j)) \
                    and (not cg_new.is_definite_unshielded_collider(j, i, k) or not cg_new.is_definite_unshielded_collider(k, i, j))\
                    and cg_new.is_undirected_from_to(i, l):
                cg_new.add_edge(i, l, -1, 1)
                Loop = True

    return cg_new

#######################################################################################################################
