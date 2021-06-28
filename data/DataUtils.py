#!/usr/bin/env python3

from data.DataSet import ContinuousDataSet
from data.DataSet import DiscreteDataSet
from data.DataSet import MixedDataSet

import numpy as np
import pandas as pd

class DataUtils():

    def load_continuous_data(self, filename, **kwargs):

        missing = '*'
        header = True
        comments = "\""
        set_delimiter = False
        delimiter = 'whitespace'

        for key, value in kwargs.items():
            if key == 'missing':
                missing = value
            elif key == 'header':
                header = bool(value)
            elif key == 'comments':
                comments = value
            elif key == 'delimiter':
                set_delimiter = True
                delimiter = value

        if set_delimiter:
            if header:
                data = np.genfromtxt(filename, skip_header=1, missing_values=missing, delimiter=delimiter, comments=comments)
                fp = open(filename, 'r')
                line = fp.readline()
                names = line.split(delimiter)
                fp.close()
            else:
                data = np.genfromtxt(filename, missing_values=missing, delimiter=delimiter, comments=comments)
                num_rows, num_columns = data.shape
                names = ['V'+ str(i) for i in range(num_columns)]
        else:
            if header:
                data = np.genfromtxt(filename, skip_header=1, missing_values=missing, comments=comments)
                fp = open(filename, 'r')
                line = fp.readline()
                names = line.split()
                fp.close()
            else:
                data = np.genfromtxt(filename, missing_values=missing, comments=comments)
                names = ['V'+ str(i) for i in range(num_columns)]

        data = ContinuousDataSet(data, names)

        return data

    def load_discrete_data(self, filename, **kwargs):

        missing = '*'
        header = True
        comments = "\""
        set_delimiter = False
        delimiter = 'whitespace'

        for key, value in kwargs.items():
            if key == 'missing':
                missing = value
            elif key == 'header':
                header = bool(value)
            elif key == 'comments':
                comments = value
            elif key == 'delimiter':
                set_delimiter = True
                delimiter = value

        if set_delimiter:
            if header:
                data = np.genfromtxt(filename, skip_header=1, missing_values=missing, delimiter=delimiter, comments=comments, dtype=str)
                fp = open(filename, 'r')
                line = fp.readline()
                names = line.split(delimiter)
                fp.close()
            else:
                data = np.genfromtxt(filename, missing_values=missing, delimiter=delimiter, comments=comments, dtype=str)
                num_rows, num_columns = data.shape
                names = ['V'+ str(i) for i in range(num_columns)]
        else:
            if header:
                data = np.genfromtxt(filename, skip_header=1, missing_values=missing, comments=comments, dtype=str)
                fp = open(filename, 'r')
                line = fp.readline()
                names = line.split()
                fp.close()
            else:
                data = np.genfromtxt(filename, missing_values=missing, comments=comments, dtype=str)
                num_rows, num_columns = data.shape
                names = ['V'+ str(i) for i in range(num_columns)]

        data = DiscreteDataSet(data, names)

        return data

    def load_mixed_data(self, filename, max_discrete, **kwargs):
        missing = '*'
        header = False
        comments = "\""
        delimiter = '\t'


        for key, value in kwargs.items():
            if key == 'missing':
                missing = value
            elif key == 'header':
                header = bool(value)
            elif key == 'comments':
                comments = value
            elif key == 'delimiter':
                delimiter = value

        if header:
            data = pd.read_csv(filename, delimiter=delimiter, comment=comments)
            rows = len(data.index)
            columns = len(data.columns)
            fp = open(filename, 'r')
            line = fp.readline()
            names = line.split(delimiter)
            fp.close()

        else:
            data = np.genfromtxt(filename, missing_values=missing, delimiter=delimiter, comments=comments, dtype=object)
            rows = len(data.index)
            columns = len(data.columns)
            names = ['V'+ str(i) for i in range(columns)]
            for i in range(columns):
                if np.unique(data[:, i]).size > max_discrete:
                    for j in range(rows):
                        data[i, j] = float(data[i, j])

        for i in range(columns):
            for j in range(rows):
                if data.iat[i, j] == missing:
                    data.iat[i, j] = None

        for i in range(columns):
            if len(data.iloc[:, i].unique()) <= max_discrete:
                for j in range(rows):
                    data.iloc[j, i] = str(data.iloc[j, i])

        data = MixedDataSet(data, names)

        return data
