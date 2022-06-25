import sys

sys.path.append("")

import io
import unittest
from itertools import product

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.Granger.Granger import Granger
from causallearn.utils.cit import fisherz
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.TimeseriesVisualization import plot_time_series


class testGraphVisualization(unittest.TestCase):

    def test_draw_CPDAG(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg = pc(data, 0.05, fisherz, True, 0,
                -1)  # Run PC and obtain the estimated graph (CausalGraph object)
        pyd = GraphUtils.to_pydot(cg.G)
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        print('finish')

    def test_draw_DAG(self):
        nodes = []
        for i in range(3):
            nodes.append(GraphNode(f"X{i + 1}"))
        dag1 = Dag(nodes)

        dag1.add_directed_edge(nodes[0], nodes[1])
        dag1.add_directed_edge(nodes[0], nodes[2])
        dag1.add_directed_edge(nodes[1], nodes[2])

        pyd = GraphUtils.to_pydot(dag1)
        pyd.write_png('dag.png')

    def test_draw_PAG(self):
        nodes = []
        for i in range(5):
            nodes.append(GraphNode(str(i)))
        dag = Dag(nodes)
        dag.add_directed_edge(nodes[0], nodes[1])
        dag.add_directed_edge(nodes[0], nodes[2])
        dag.add_directed_edge(nodes[1], nodes[3])
        dag.add_directed_edge(nodes[2], nodes[4])
        dag.add_directed_edge(nodes[3], nodes[4])
        pag = dag2pag(dag, [nodes[0], nodes[2]])

        pyd = GraphUtils.to_pydot(pag)
        pyd.write_png('pag.png')

    def test_plot_simple_time_series_graph(self):
        coef_matrix = np.zeros((3, 3, 4))
        coef_matrix[0, 0, 1] = 1
        coef_matrix[1, 1, 1] = 1
        coef_matrix[2, 2, 1] = 1
        coef_matrix[0, 1, 1] = 1
        coef_matrix[1, 0, 1] = 1

        plot_time_series(coef_matrix=coef_matrix)

    def test_plot_granger(self):
        df = pd.read_csv('https://cdn.jsdelivr.net/gh/selva86/datasets/a10.csv', parse_dates=['date'])
        df['month'] = df.date.dt.month
        dataset = df[['value', 'month']].to_numpy()
        maxlag = 2
        G = Granger(maxlag=maxlag)
        coeff = G.granger_lasso(data=dataset)
        dim = dataset.shape[1]
        coef_matrix = np.zeros((dataset.shape[1], dataset.shape[1], maxlag + 1))
        for i, j, tau in product(range(dim), range(dim), range(1, maxlag + 1)):
            coef_matrix[i, j, tau] = coeff[j, (tau - 1) * dim + i]
        plot_time_series(coef_matrix=coef_matrix)

    def test_color(self):
        nodes = []
        for i in range(3):
            nodes.append(GraphNode(f"X{i + 1}"))
        pag = GeneralGraph(nodes)
        edge = Edge(nodes[0], nodes[1], Endpoint.TAIL, Endpoint.ARROW)
        edge.properties.append(Edge.Property.dd)
        pag.add_edge(edge)
        edges = [edge]
        pyd = GraphUtils.to_pydot(pag, edges)
        pyd.write_png('green.png')

    def test_plot_with_labels(self):
        nodes = []
        for i in range(3):
            nodes.append(GraphNode(f"X{i + 1}"))
        dag1 = Dag(nodes)

        dag1.add_directed_edge(nodes[0], nodes[1])
        dag1.add_directed_edge(nodes[0], nodes[2])
        dag1.add_directed_edge(nodes[1], nodes[2])

        pyd = GraphUtils.to_pydot(dag1, labels=["A", "B", "C"])
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def test_draw_graph_with_labels(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        cg = pc(data, 0.05, fisherz, True, 0, -1)
        cg.draw_pydot_graph(labels=[f"Node_{i + 1}" for i in range(data.shape[1])])
