import unittest

import numpy as np

from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, mv_fisherz
from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge


class TestPC(unittest.TestCase):

    def test_forbidden_by_node(self):
        node1 = GraphNode('spam')
        node2 = GraphNode('ham')

        bk = BackgroundKnowledge().add_forbidden_by_node(node1, node2)

        assert bk.forbidden_rules_specs.__contains__((node1, node2))
        assert bk.is_forbidden(node1, node2)

        bk.remove_forbidden_by_node(node1, node2)

        assert not bk.forbidden_rules_specs.__contains__((node1, node2))
        assert not bk.is_forbidden(node1, node2)

    def test_required_by_node(self):
        node1 = GraphNode('spam')
        node2 = GraphNode('ham')

        bk = BackgroundKnowledge().add_required_by_node(node1, node2)

        assert bk.required_rules_specs.__contains__((node1, node2))
        assert bk.is_required(node1, node2)

        bk.remove_required_by_node(node1, node2)

        assert not bk.required_rules_specs.__contains__((node1, node2))
        assert not bk.is_required(node1, node2)

    def test_forbidden_by_pattern(self):
        node1 = GraphNode('spam')
        node2 = GraphNode('ham')
        node_pattern1 = '^s.*m$'
        node_pattern2 = node2.get_name()

        bk = BackgroundKnowledge().add_forbidden_by_pattern(node_pattern1, node_pattern2)

        assert bk.forbidden_pattern_rules_specs.__contains__((node_pattern1, node_pattern2))
        assert bk.is_forbidden(node1, node2)

        bk.remove_forbidden_by_pattern(node_pattern1, node_pattern2)

        assert not bk.forbidden_pattern_rules_specs.__contains__((node_pattern1, node_pattern2))
        assert not bk.is_forbidden(node1, node2)

    def test_required_by_pattern(self):
        node1 = GraphNode('spam')
        node2 = GraphNode('ham')
        node_pattern1 = '^s.*m$'
        node_pattern2 = node2.get_name()

        bk = BackgroundKnowledge().add_required_by_pattern(node_pattern1, node_pattern2)

        assert bk.required_pattern_rules_specs.__contains__((node_pattern1, node_pattern2))
        assert bk.is_required(node1, node2)

        bk.remove_required_by_pattern(node_pattern1, node_pattern2)

        assert not bk.required_pattern_rules_specs.__contains__((node_pattern1, node_pattern2))
        assert not bk.is_required(node1, node2)

    def test_add_node_to_tier(self):
        node1 = GraphNode('spam')
        node2 = GraphNode('ham')

        bk = BackgroundKnowledge().add_node_to_tier(node1, 1).add_node_to_tier(node2, 2)

        assert bk.tier_map.get(1).__contains__(node1)
        assert bk.tier_map.get(2).__contains__(node2)
        assert bk.tier_value_map[node1] == 1
        assert bk.tier_value_map[node2] == 2
        assert bk.is_forbidden(node2, node1)

        bk.remove_node_from_tier(node1, 1)

        assert not bk.tier_map.get(1).__contains__(node1)
        assert not bk.tier_value_map.keys().__contains__(node1)
        assert not bk.is_forbidden(node1, node2)

    def test_orient_by_background_knowledge(self):
        cg = CausalGraph(4)
        nodes = cg.G.get_nodes()

        assert cg.G.is_undirected_from_to(nodes[0], nodes[1])
        assert cg.G.is_undirected_from_to(nodes[2], nodes[0])
        assert cg.G.is_undirected_from_to(nodes[1], nodes[2])
        assert cg.G.is_undirected_from_to(nodes[3], nodes[1])

        bk = BackgroundKnowledge() \
            .add_forbidden_by_node(nodes[0], nodes[1]) \
            .add_forbidden_by_node(nodes[2], nodes[0]) \
            .add_required_by_node(nodes[1], nodes[2]) \
            .add_required_by_node(nodes[3], nodes[1])

        orient_by_background_knowledge(cg, bk)

        assert cg.G.is_directed_from_to(nodes[1], nodes[0])
        assert cg.G.is_directed_from_to(nodes[0], nodes[2])
        assert cg.G.is_directed_from_to(nodes[1], nodes[2])
        assert cg.G.is_directed_from_to(nodes[3], nodes[1])

    def test_skeleton_discovery(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg_1 = SkeletonDiscovery.skeleton_discovery(data, 0.05, fisherz, True, background_knowledge=None)
        assert cg_1.G.is_undirected_from_to(cg_1.G.nodes[0], cg_1.G.nodes[3])

        bk = BackgroundKnowledge() \
            .add_forbidden_by_node(cg_1.G.nodes[0], cg_1.G.nodes[3]) \
            .add_forbidden_by_node(cg_1.G.nodes[3], cg_1.G.nodes[0])
        cg_1 = SkeletonDiscovery.skeleton_discovery(data, 0.05, fisherz, True, background_knowledge=bk)
        assert cg_1.G.get_edge(cg_1.G.nodes[0], cg_1.G.nodes[3]) is None

    def test_pc_with_background_knowledge(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg_without_background_knowledge = pc(data, 0.05, fisherz, True, 0,
                                             0)  # Run PC and obtain the estimated graph (CausalGraph object)
        nodes = cg_without_background_knowledge.G.get_nodes()

        assert cg_without_background_knowledge.G.is_directed_from_to(nodes[2], nodes[8])
        assert cg_without_background_knowledge.G.is_undirected_from_to(nodes[7], nodes[17])

        bk = BackgroundKnowledge() \
            .add_forbidden_by_node(nodes[2], nodes[8]) \
            .add_forbidden_by_node(nodes[8], nodes[2]) \
            .add_required_by_node(nodes[7], nodes[17])
        cg_with_background_knowledge = pc(data, 0.05, fisherz, True, 0, 0, background_knowledge=bk)

        assert cg_with_background_knowledge.G.get_edge(nodes[2], nodes[8]) is None
        assert cg_with_background_knowledge.G.is_directed_from_to(nodes[7], nodes[17])

    def test_mvpc_with_background_knowledge(self):
        data_path = "data_linear_10.txt"
        data = np.loadtxt(data_path, skiprows=1)  # Import the file at data_path as data
        cg_without_background_knowledge = pc(data, 0.05, mv_fisherz, True, 0,
                                             0, mvpc=True)  # Run PC and obtain the estimated graph (CausalGraph object)

        nodes = cg_without_background_knowledge.G.get_nodes()

        assert cg_without_background_knowledge.G.is_directed_from_to(nodes[0], nodes[3])
        assert cg_without_background_knowledge.G.is_directed_from_to(nodes[0], nodes[12])

        bk = BackgroundKnowledge() \
            .add_forbidden_by_node(nodes[0], nodes[3]) \
            .add_forbidden_by_node(nodes[3], nodes[0]) \
            .add_required_by_node(nodes[7], nodes[17])
        cg_with_background_knowledge = pc(data, 0.05, mv_fisherz, True, 0, 0, mvpc=True, background_knowledge=bk)

        assert cg_with_background_knowledge.G.get_edge(nodes[0], nodes[3]) is None
        assert cg_with_background_knowledge.G.is_directed_from_to(nodes[0], nodes[12])


