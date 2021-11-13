from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


def txt2generalgraph(filename):
    G = GeneralGraph([])
    node_map = {}
    with open(filename, "r") as file:
        next_nodes_line = False
        for line in file.readlines():
            line = line.strip()
            words = line.split()
            if len(words) > 1 and words[1] == 'Nodes:':
                next_nodes_line = True
            elif len(line) > 0 and next_nodes_line:
                next_nodes_line = False
                nodes = line.split(';')
                # print(nodes)
                for node in nodes:
                    node_map[node] = GraphNode(node)
                    G.add_node(node_map[node])
            elif len(words) > 0 and words[0][-1] == '.':
                next_nodes_line = False
                node1 = words[1]
                node2 = words[3]
                end1 = words[2][0]
                end2 = words[2][-1]
                if end1 == '<':
                    end1 = '>'
                end1 = to_endpoint(end1)
                end2 = to_endpoint(end2)
                edge = Edge(node_map[node1], node_map[node2], Endpoint.CIRCLE, Endpoint.CIRCLE)
                mod_endpoint(edge, node_map[node1], end1)
                mod_endpoint(edge, node_map[node2], end2)
                G.add_edge(edge)
    return G


def to_endpoint(s):
    if s == 'o':
        return Endpoint.CIRCLE
    elif s == '>':
        return Endpoint.ARROW
    elif s == '-':
        return Endpoint.TAIL
    else:
        raise NotImplementedError


def mod_endpoint(edge, z, end):
    if edge.get_node1() == z:
        edge.set_endpoint1(end)
    elif edge.get_node2() == z:
        edge.set_endpoint2(end)
    else:
        raise ValueError("z not in edge")
