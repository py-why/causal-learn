#!/usr/bin/env python3

from graph.GraphNode import GraphNode

import pandas as pd
import numpy as np

class DiscreteDataSet(DataSet):

    def __init__(self, data, var_names):

        self.data = data

        nodes = []
        names = []
        for name in var_names:
            nodes.append(GraphNode(name))

        self.variables = nodes
        self.variable_names = var_names

        self.num_rows = len(data.index)
        self.num_vars = data.columns

    def get_data(self):
        return self.dataset

    def get_variables(self):
        return self.variables

    def get_variable_names(self):
        return self.variable_names

    def get_num_rows(self):
        return self.num_rows

    def get_num_variables(self):
        return self.num_vars

    def get_column(self, variable):
        return self.variable_map[variable]

    def get_double(self, row, column):

        val = self.data[row, column]

        if val.dtype == pandas.float64:
            return val
        else:
            return None

    # Currently returns None for non-number values. May return ordering once
    # infrastructure has been built for that
    def get_int(self, row, column):

        if self.data.iloc[:, column].dtype = dtype('int64'):
            return = self.data[row, column]
        else:
            return None

    def get_object(self, row, column):
        return self.data[row, column]

    def get_var_by_index(self, column):
        return self.variables[column]

    def get_var_by_name(self, var_name):
        return self.variables[self.variable_names.index(var_name)]

    def is_continuous(self):
        return False

    def is_discrete(self):
        return False

    def is_mixed(self):
        return True

    def remove_col_by_index(self, index):
        self.data = np.delete(self.data, index, 1)

    def remove_col_by_name(self, name):
        self.data = np.delete(self.data, self.variable_names.index(var_name), 1)

    def remove_columns(self, columns):
        self.data = np.delete(self.data, columns, 1)

    def remove_rows(self, rows):
        self.data = np.delete(self.data, rows, 0)


    def set_int(self, row, column, value):

        data = self.data
        data[row, column] == str(value)
        self.data = data

    def set_object(self, row, column, value):

        data = self.data
        data[row, column] == str(value)
        self.data = data

    def subset_cols_by_index(self, indices):
        self.data = numpy.take(self.data, indices, 1)

    def subset_cols_by_variable(self, variables):
        indices = []

        indices.append(self.variables.index(variable) for variable in variables)

    def subset_rows(self, indices):
        self.data = numpy.take(self.data, indices, 0)

    def permute_rows(self):
        self.data = numpy.random.permutation(self.data)

    def __str__(self):
        s = ""

        for name in self.variable_names:
            s = s + name + '\t'

        s = s + '\n'

        for i in range(self.num_rows):
            for j in range(self.num_vars):
                s = s + str(self.data[i, j])

                if j < self.num_vars - 1:
                    s = s + '\t'
            if i < self.num_rows - 1:
                s = s + '\n'

        return s
