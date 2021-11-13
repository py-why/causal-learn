import re

from causallearn.graph.Node import Node


class BackgroundKnowledge(object):
    def __init__(self):
        self.forbidden_rules_specs = set()
        self.forbidden_pattern_rules_specs = set()
        self.required_rules_specs = set()
        self.required_pattern_rules_specs = set()
        self.tier_map = {}
        self.tier_value_map = {}

    def add_forbidden_by_node(self, node1, node2):
        """
        Marks the edge node1 --> node2 as forbidden.

        Parameters
        ----------
        node1: the from node which the edge is forbidden
        node2: the end node which the edge is forbidden

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError(
                'node must not be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(type(node2)))

        self.forbidden_rules_specs.add((node1, node2))

        return self

    def add_required_by_node(self, node1, node2):
        """
        Marks the edge node1 --> node2 as required.

        Parameters
        ----------
        node1: the from node which the edge is required
        node2: the end node which the edge is required

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError(
                'node must not be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(type(node2)))

        self.required_rules_specs.add((node1, node2))

        return self

    def add_forbidden_by_pattern(self, node_pattern1, node_pattern2):
        """
        Marks the edges node_pattern1 --> node_pattern2 as forbidden.

        Parameters
        ----------
        node_pattern1: the regular expression of the name of the from node which the edge is forbidden.
        node_pattern2: the regular expression of the name of the end node which the edge is forbidden.

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if type(node_pattern1) != str or type(node_pattern2) != str:
            raise TypeError('node_pattern must be type of str. node_pattern1 = ' + str(
                type(node_pattern1)) + ' node_pattern2 = ' + str(type(node_pattern2)))

        self.forbidden_pattern_rules_specs.add((node_pattern1, node_pattern2))

        return self

    def add_required_by_pattern(self, node_pattern1, node_pattern2):
        """
        Marks the edges node_pattern1 --> node_pattern2 as required.

        Parameters
        ----------
        node_pattern1: the regular expression of the name of the from node which the edge is required.
        node_pattern2: the regular expression of the name of the end node which the edge is required.

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if type(node_pattern1) != str or type(node_pattern2) != str:
            raise TypeError('node_pattern must be type of str. node_pattern1 = ' + str(
                type(node_pattern1)) + ' node_pattern2 = ' + str(type(node_pattern2)))

        self.required_pattern_rules_specs.add((node_pattern1, node_pattern2))

        return self

    def _ensure_tiers(self, tier):
        if type(tier) != int:
            raise TypeError('tier must be int type. tier = ' + str(type(tier)))

        for t in range(tier + 1):
            if not self.tier_map.keys().__contains__(t):
                self.tier_map[t] = set()

    def add_node_to_tier(self, node, tier):
        """
        Mark the tier of the node. And the edges from the equal or higher tiers to the other tiers are forbidden.

        Parameters
        ----------
        node: Node type variable
        tier: the tier of node, which is a non-negative integer.

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if (not isinstance(node, Node)) or type(tier) != int:
            raise TypeError(
                'node must be instance of Node. tier must be int type. node = ' + str(type(node)) + ' tier = ' + str(
                    type(tier)))
        if tier < 0:
            raise TypeError('tier must be a non-negative integer. tier = ' + str(tier))

        self._ensure_tiers(tier)
        self.tier_map.get(tier).add(node)
        self.tier_value_map[node] = tier

        return self

    def _is_node_match_regular_expression(self, pattern, node):
        if re.match(pattern, node.get_name()) is not None:
            return True
        else:
            return False

    def is_forbidden(self, node1, node2):
        """
        check whether the edge node1 --> node2 is forbidden

        Parameters
        ----------
        node1: the from node in edge which is checked
        node2: the to node in edge which is checked

        Returns
        -------
        if the  edge node1 --> node2 is forbidden, then return True, otherwise False.
        """
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError('node1 and node2 must be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(
                type(node2)))

        # first check in forbidden_rules_specs
        for (from_node, to_node) in self.forbidden_rules_specs:
            if from_node == node1 and to_node == node2:
                return True

        # then check in forbidden_pattern_rules_specs
        for (from_node_pattern, to_node_pattern) in self.forbidden_pattern_rules_specs:
            if self._is_node_match_regular_expression(from_node_pattern,
                                                      node1) and self._is_node_match_regular_expression(to_node_pattern,
                                                                                                        node2):
                return True

        # then check in tier_map
        if self.tier_value_map.keys().__contains__(node1) and self.tier_value_map.keys().__contains__(node2):
            if self.tier_value_map.get(node1) >= self.tier_value_map.get(node2):
                return True

        return False

    def is_required(self, node1, node2):
        """
        check whether the edge node1 --> node2 is required

        Parameters
        ----------
        node1: the from node in edge which is checked
        node2: the to node in edge which is checked

        Returns
        -------
        if the  edge node1 --> node2 is required, then return True, otherwise False.
        """
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError('node1 and node2 must be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(
                type(node2)))

        # first check in required_rules_specs
        for (from_node, to_node) in self.required_rules_specs:
            if from_node == node1 and to_node == node2:
                return True

        # then check in required_pattern_rules_specs
        for (from_node_pattern, to_node_pattern) in self.required_pattern_rules_specs:
            if self._is_node_match_regular_expression(from_node_pattern,
                                                      node1) and self._is_node_match_regular_expression(to_node_pattern,
                                                                                                        node2):
                return True

        return False

    def remove_forbidden_by_node(self, node1, node2):
        """
        remove the forbidden mark of the edge node1 --> node2.

        Parameters
        ----------
        node1: the from node which the edge is used to be forbidden
        node2: the end node which the edge is used to be forbidden

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError(
                'node must not be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(type(node2)))

        if self.forbidden_rules_specs.__contains__((node1, node2)):
            self.forbidden_rules_specs.remove((node1, node2))

        return self

    def remove_required_by_node(self, node1, node2):
        """
        remove the required mark of the edge node1 --> node2.

        Parameters
        ----------
        node1: the from node which the edge is used to be required
        node2: the end node which the edge is used to be required

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError(
                'node must not be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(type(node2)))

        if self.required_rules_specs.__contains__((node1, node2)):
            self.required_rules_specs.remove((node1, node2))

        return self

    def remove_forbidden_by_pattern(self, node_pattern1, node_pattern2):
        """
        remove the forbidden mark of the edges node_pattern1 --> node_pattern2.

        Parameters
        ----------
        node_pattern1: the regular expression of the name of the from node which the edge is used to be forbidden.
        node_pattern2: the regular expression of the name of the end node which the edge is used to be forbidden.

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if type(node_pattern1) != str or type(node_pattern2) != str:
            raise TypeError('node_pattern must be type of str. node_pattern1 = ' + str(
                type(node_pattern1)) + ' node_pattern2 = ' + str(type(node_pattern2)))

        if self.forbidden_pattern_rules_specs.__contains__((node_pattern1, node_pattern2)):
            self.forbidden_pattern_rules_specs.remove((node_pattern1, node_pattern2))

        return self

    def remove_required_by_pattern(self, node_pattern1, node_pattern2):
        """
        remove the required mark of the edges node_pattern1 --> node_pattern2.

        Parameters
        ----------
        node_pattern1: the regular expression of the name of the from node which the edge is used to be required.
        node_pattern2: the regular expression of the name of the end node which the edge is used to be required.

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if type(node_pattern1) != str or type(node_pattern2) != str:
            raise TypeError('node_pattern must be type of str. node_pattern1 = ' + str(
                type(node_pattern1)) + ' node_pattern2 = ' + str(type(node_pattern2)))

        if self.required_pattern_rules_specs.__contains__((node_pattern1, node_pattern2)):
            self.required_pattern_rules_specs.remove((node_pattern1, node_pattern2))

        return self

    def remove_node_from_tier(self, node, tier):
        """
        remove the mark of the tier of the node.

        Parameters
        ----------
        node: Node type variable
        tier: the used tier of node.

        Returns
        -------
        The object itself, which is for the convenience of construction.
        """
        if (not isinstance(node, Node)) or type(tier) != int:
            raise TypeError(
                'node must be instance of Node. tier must be int type. node = ' + str(type(node)) + ' tier = ' + str(
                    type(tier)))
        if tier < 0:
            raise TypeError('tier must be a non-negative integer. tier = ' + str(tier))

        self._ensure_tiers(tier)
        if self.tier_map.get(tier).__contains__(node):
            self.tier_map.get(tier).remove(node)
        if self.tier_value_map.keys().__contains__(node):
            self.tier_value_map.pop(node)

        return self
