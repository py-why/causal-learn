from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def orient_by_background_knowledge(cg, background_knowledge):
    """
    orient the direction of edges using background background_knowledge after running skeleton_discovery in PC algorithm

    Parameters
    ----------
    cg: a CausalGraph object. Where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
    background_knowledge: artificial background background_knowledge

    Returns
    -------

    """
    if type(cg) != CausalGraph or type(background_knowledge) != BackgroundKnowledge:
        raise TypeError(
            'cg must be type of CausalGraph and background_knowledge must be type of BackgroundKnowledge. cg = ' + str(
                type(cg)) + ' background_knowledge = ' + str(type(background_knowledge)))
    for edge in cg.G.get_graph_edges():
        if cg.G.is_undirected_from_to(edge.get_node1(), edge.get_node2()):
            if background_knowledge.is_forbidden(edge.get_node2(), edge.get_node1()):
                cg.G.remove_edge(edge)
                cg.G.add_directed_edge(edge.get_node1(), edge.get_node2())
            elif background_knowledge.is_forbidden(edge.get_node1(), edge.get_node2()):
                cg.G.remove_edge(edge)
                cg.G.add_directed_edge(edge.get_node2(), edge.get_node1())
            elif background_knowledge.is_required(edge.get_node2(), edge.get_node1()):
                cg.G.remove_edge(edge)
                cg.G.add_directed_edge(edge.get_node2(), edge.get_node1())
            elif background_knowledge.is_required(edge.get_node1(), edge.get_node2()):
                cg.G.remove_edge(edge)
                cg.G.add_directed_edge(edge.get_node1(), edge.get_node2())
