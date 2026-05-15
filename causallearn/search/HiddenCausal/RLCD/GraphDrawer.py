import pydot
import json


PARAMS = {
    "default_node_colour": "black",
    "refined_node_colour": "red",
    "default_edge_colour": "black",
    "directed_edge_colour": "black",
}
#with open("params.json") as f:
#    PARAMS.update(json.load(f))


class DotGraph:
    """
    A class used to construct a directed/undirected graph and parses the graph
    into a .dot file for pydot plotting.
    """

    def __init__(
        self,
        default_node_colour: str = PARAMS["default_node_colour"],
        refined_node_colour: str = PARAMS["refined_node_colour"],
        default_edge_colour: str = PARAMS["default_edge_colour"],
        directed_edge_colour: str = PARAMS["directed_edge_colour"],
    ):
        self.default_node_colour = default_node_colour
        self.refined_node_colour = refined_node_colour
        self.default_edge_colour = default_edge_colour
        self.directed_edge_colour = directed_edge_colour
        self.nodes = set()
        self.dirEdges = set()
        self.undirEdges = set()
        self.nodecolor = {}

    def addNode(self, V, refined=False):
        self.nodes.add(V)
        if refined:
            self.nodecolor[V] = self.refined_node_colour
        else:
            self.nodecolor[V] = self.default_node_colour

    def addNodeByColor(self, V, Color):
        self.nodes.add(V)
        self.nodecolor[V] = Color

    # 0 for undirected, 1 for directed
    def addEdge(self, u, v, type=0):
        if type == 0:
            self.undirEdges.add(frozenset([u, v]))

        elif type == 1:
            self.dirEdges.add((u, v))

    def edges(self, V, type=0):
        edgelist = []
        if type == 0:
            for edge in self.undirEdges:
                if V in edge:
                    edgelist.append(edge)

        if type == 1:
            for edge in self.dirEdges:
                if V in edge:
                    edgelist.append(edge)
        return edgelist

    def removeUndirEdgesFromNode(self, V):
        edgesToRemove = set()
        for edgeSet in self.undirEdges:
            if V in edgeSet:
                edgesToRemove.add(edgeSet)
        self.undirEdges = self.undirEdges - edgesToRemove

    # def toDot(self, outpath: str):
    #     # TODO: Improve plotting for phase III with undirected edges
    #     text = "digraph {\n"

    #     # Add nodes
    #     for node in self.nodes:
    #         text += f"{node} [color = {self.nodecolor[node]}]; "
    #     text += "\n"

    #     # Add undirected edges
    #     text += "subgraph Undirected {\n"
    #     text += f"edge [dir=none, color={self.default_edge_colour}]\n"
    #     for edgeSet in self.undirEdges:
    #         edgeSet = list(edgeSet)
    #         text += f"{edgeSet[0]} -> {edgeSet[1]}\n"

    #     text += "}\n\n"

    #     # Add directed Edges
    #     text += "subgraph Directed {\n"
    #     text += f"edge [color={self.directed_edge_colour}]\n"
    #     for edgeSet in self.dirEdges:
    #         edgeSet = list(edgeSet)
    #         text += f"{edgeSet[0]} -> {edgeSet[1]}\n"

    #     text += "}\n\n"
    #     text += "}\n"
    #     with open(outpath, "w") as f:
    #         f.write(text)

    def toDot(self, outpath: str):
        # UTF-8 support
        text = "digraph {\n"
        text += 'charset="UTF-8";\n'
        
        # Add nodes - ensure node names are quoted to handle special characters
        for node in self.nodes:
            text += f'"{node}" [color = {self.nodecolor[node]}];\n'
        text += "\n"

        # Add undirected edges
        text += "subgraph Undirected {\n"
        text += f"edge [dir=none, color={self.default_edge_colour}]\n"
        for edgeSet in self.undirEdges:
            edgeSet = list(edgeSet)
            text += f'"{edgeSet[0]}" -> "{edgeSet[1]}"\n'

        text += "}\n\n"

        # Add directed Edges
        text += "subgraph Directed {\n"
        text += f"edge [color={self.directed_edge_colour}]\n"
        for edgeSet in self.dirEdges:
            edgeSet = list(edgeSet)
            text += f'"{edgeSet[0]}" -> "{edgeSet[1]}"\n'

        text += "}\n\n"
        text += "}\n"
        
        # 指定 UTF-8 编码写入文件
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(text)


def printGraph(O: object, outpath="plots/test.png", layout="dot", res=100):
    """
    Function to plot a graph object using pydot from various types of graph
    objects.
    """
    if isinstance(O, DotGraph):
        dotGraph = O
    else:
        try:
            dotGraph = O.getDotGraph()
        except:
            raise TypeError(f"{O} is of type {type(O)}, not supported.")
        
    dot_path = outpath.replace(".png", ".dot")

    dotGraph.toDot(dot_path)
    graphs = pydot.graph_from_dot_file(dot_path)
    graphs[0].set_size(f'"{res},{res}!"')
    graphs[0].set_layout(layout)
    graphs[0].write_png(outpath)


def AdjToGraph(Adj, varnames_for_Adj):

    dotGraph = DotGraph()
    for x in varnames_for_Adj:
        if x.startswith("L"):
            dotGraph.addNodeByColor(x, 'red')
        else:
            dotGraph.addNodeByColor(x, 'blue')

    for i in range(len(Adj)):
        for j in range(i+1, len(Adj)):
            
            if (Adj[i,j]==1 and Adj[j,i]==1) or Adj[i,j]==-1 and Adj[j,i]==-1:
                dotGraph.addEdge(varnames_for_Adj[i], varnames_for_Adj[j])
            elif Adj[i,j]==-1 and Adj[j,i]==1:
                dotGraph.addEdge(varnames_for_Adj[i], varnames_for_Adj[j], type=1)
            elif Adj[j,i]==-1 and Adj[i,j]==1:
                dotGraph.addEdge(varnames_for_Adj[j], varnames_for_Adj[i], type=1)

    return dotGraph