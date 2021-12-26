from itertools import chain, combinations

import numpy as np

from causallearn.utils.cit import (chisq, chisq_notoptimized, gsq,
                                   gsq_notoptimized)


def test_new_old_gsq_chisq_equivalent(self):
    def powerset(iterable):
        return chain.from_iterable(combinations(list(iterable), r) for r in range(len(iterable) + 1))

    def _unique(column):
        return np.unique(column, return_inverse=True)[1]

    data_path = "data_discrete_10.txt"
    data = np.loadtxt(data_path, skiprows=1)
    data = np.apply_along_axis(_unique, 0, data).astype(np.int32)
    cardinalities = np.max(data, axis=0) + 1

    for X in range(data.shape[1]):
        for Y in range(X + 1, data.shape[1]):
            for S in powerset([_ for _ in range(data.shape[1]) if _ != X and _ != Y]):
                assert np.isclose(gsq(data, X, Y, S, cardinalities), gsq_notoptimized(data, X, Y, S))
                assert np.isclose(chisq(data, X, Y, S, cardinalities), chisq_notoptimized(data, X, Y, S))
                print(f'{X};{Y}|{S} passed')
