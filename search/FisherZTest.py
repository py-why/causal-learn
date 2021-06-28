#!/usr/bin/env python3

from copy import deepcopy
from itertools import combinations, chain, permutations
from math import sqrt, log
from scipy.stats import norm, chi2
from search import IndependenceTest
from data import DataSet

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings

class FisherZTest(IndependenceTest):

    def __init__(self, data, alpha):

        if not data.datatype == 'continuous':
            raise TypeError("FizherZ can only be performed on continuous data")

        self.data = data
        self.corr = data.get_correlation_matrix()
        self.variables = data.get_variables()
        self.num_vars = data.get_num_variables()
        self.variable_names = data.get_variable_names()
        self.num_rows = data.get_num_rows()

        node_map = {}

        for i in range(self.num_vars):
            node = self.variables[i]
            node_map[node] = i

        self.variable_map = node_map

        name_map = {}

        for i in range(self.num_vars):
            name = self.variable_names[i]
            name_map[name] = i

        self.name_map = name_map

        self.alpha = alpha

    def is_independent(node_x, node_y, z_list):

        x = self.variable_map{node_x}
        y = self.variable_map{node_y}

        z = []
        for variable in z_list:
            z.append(self.variable_map{variable})

        var = list((x, y) + z)
        sub_corr_matrix = self.corr[np.ix_(var, var)]
        inv = np.linalg.inv(sub_corr_matrix)
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.num_rows - len(z) - 3) * abs(Z)
        # p = 2 * (1 - norm.cdf(abs(X)))
        p = 1 - norm.cdf(abs(X))
        return p > self.alpha
