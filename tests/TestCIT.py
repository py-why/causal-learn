# from itertools import chain, combinations
import unittest

import numpy as np
from causallearn.utils.cit import CIT
# from causallearn.utils.cit import (chisq, chisq_notoptimized, gsq, gsq_notoptimized)


class TestCIT(unittest.TestCase):
    def test_fisherz_singularity_problem(self):
        X1 = X2 = np.random.normal(size=1000)
        X = np.array([X1, X2])

        cit = CIT(data=X, method='fisherz')

        try:
            cit.fisherz(0, 1, tuple())
        except ValueError:
            print('Catch Singularity Problem')
            return

        assert False

# def test_new_old_gsq_chisq_equivalent(self):
#     def powerset(iterable):
#         return chain.from_iterable(combinations(list(iterable), r) for r in range(len(iterable) + 1))
#
#     def _unique(column):
#         return np.unique(column, return_inverse=True)[1]
#
#     data_path = "data_discrete_10.txt"
#     data = np.loadtxt(data_path, skiprows=1)
#     data = np.apply_along_axis(_unique, 0, data).astype(np.int32)
#     cardinalities = np.max(data, axis=0) + 1
#
#     for X in range(data.shape[1]):
#         for Y in range(X + 1, data.shape[1]):
#             for S in powerset([_ for _ in range(data.shape[1]) if _ != X and _ != Y]):
#                 assert np.isclose(gsq(data, X, Y, S, cardinalities), gsq_notoptimized(data, X, Y, S))
#                 assert np.isclose(chisq(data, X, Y, S, cardinalities), chisq_notoptimized(data, X, Y, S))
#                 print(f'{X};{Y}|{S} passed')
