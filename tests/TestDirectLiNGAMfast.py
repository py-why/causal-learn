import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd
import subprocess

from causallearn.search.FCMBased import lingam

def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        print("CUDA Version found:\n", nvcc_version)
        return True
    except Exception as e:
        print("CUDA not found or nvcc not in PATH:", e)
        return False

class TestDirectLiNGAMFast(unittest.TestCase):

    def test_DirectLiNGAM(self):
        np.set_printoptions(precision=3, suppress=True)
        np.random.seed(100)
        x3 = np.random.uniform(size=1000)
        x0 = 3.0 * x3 + np.random.uniform(size=1000)
        x2 = 6.0 * x3 + np.random.uniform(size=1000)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
        x5 = 4.0 * x0 + np.random.uniform(size=1000)
        x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
        X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])

        cuda = get_cuda_version()
        if cuda:
            model = lingam.DirectLiNGAM(measure='pwling_fast')
            model.fit(X)

            print(model.causal_order_)
            print(model.adjacency_matrix_)
