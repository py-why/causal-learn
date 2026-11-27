import unittest

import numpy as np
import pandas as pd

from causallearn.utils.cit import CIT


class TestCIT(unittest.TestCase):
    def test_fisherz_singularity_problem(self):
        X1 = X2 = np.random.normal(size=1000)
        X = np.array([X1, X2]).T

        cit = CIT(data=X, method='fisherz')

        with self.assertRaises(ValueError) as context:
            cit.fisherz(0, 1, tuple())

    def test_pandas_nullable_dtypes(self):
        # DataFrames with nullable dtypes (Float64, Int64) give object arrays in numpy
        rng = np.random.default_rng(0)
        a = rng.normal(size=200)
        b = a + rng.normal(size=200)
        c = (b + rng.normal(size=200) > 0).astype(int)
        df = pd.DataFrame({"a": a, "b": b, "c": c}).astype({"a": "Float64", "b": "Float64", "c": "Int64"})
        data = df.to_numpy()
        assert data.dtype == object
        for method in ["fisherz", "chisq", "kci"]:
            expected = CIT(data=df.to_numpy(dtype=float), method=method)(0, 2, (1,))
            assert CIT(data=data, method=method)(0, 2, (1,)) == expected
