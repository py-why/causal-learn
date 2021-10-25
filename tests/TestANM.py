import sys, os
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
from pytrad.search.FCMBased.ANM.ANM import ANM
import sys
import pandas as pd
import unittest
from pickle import load
import numpy as np


class TestANM(unittest.TestCase):

    # example1
    # for data with two dimensions, granger test.
    def test_anm_pair(self):
        df = pd.read_csv('TestData/pair0001.txt', sep=' ', header=None)
        dataset = df.to_numpy()
        anm = ANM()
        n = dataset.shape[0]
        p_value_foward, p_value_backward = anm.cause_or_effect(dataset[:,0].reshape(n,1), dataset[:,1].reshape(n,1))
        print('pvalue for x->y is {:.4f}'.format(p_value_foward))
        print('pvalue for y->x is {:.4f}'.format(p_value_backward))


if __name__ == '__main__':
    test = TestANM()
    test.test_anm_pair()