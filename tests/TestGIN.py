import random
import sys
import io

sys.path.append("")
import unittest

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from causallearn.search.HiddenCausal.GIN.GIN import GIN


class TestGIN(unittest.TestCase):
    def test_case1(self):
        sample_size = 1000
        np.random.seed(0)
        L1 = np.random.uniform(-1, 1, size=sample_size) ** 5
        L2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X1 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X3 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X4 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5

        data = np.array([X1, X2, X3, X4]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        print(g, k)

        # Visualization using pydot
        from causallearn.utils.GraphUtils import GraphUtils
        pyd = GraphUtils.to_pydot(g)
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def test_case2(self):
        sample_size = 1000
        np.random.seed(0)
        L1 = np.random.uniform(-1, 1, size=sample_size) ** 5
        L2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        L3 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1,
                                                                                                     size=sample_size) ** 5
        X1 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X3 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X4 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X5 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X6 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X7 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X8 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X9 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5

        data = np.array([X1, X2, X3, X4, X5, X6, X7, X8, X9]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        print(g, k)

        # Visualization using pydot
        from causallearn.utils.GraphUtils import GraphUtils
        pyd = GraphUtils.to_pydot(g)
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def test_case3(self):
        sample_size = 1000
        random.seed(42)
        np.random.seed(42)
        L1 = np.random.uniform(-1, 1, size=sample_size) ** 5
        L2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        L3 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        L4 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5

        X1 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X3 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X4 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X5 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X6 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X7 = np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X8 = np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5

        data = np.array([X1, X2, X3, X4, X5, X6, X7, X8]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        print(g, k)

        # Visualization using pydot
        from causallearn.utils.GraphUtils import GraphUtils
        pyd = GraphUtils.to_pydot(g)
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()