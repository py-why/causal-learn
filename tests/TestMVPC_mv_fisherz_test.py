import os
import sys
import timeit
# get current directory
path = os.getcwd()

# get parent directory
path=os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(path)

import unittest
import numpy as np
from causallearn.utils.cit import fisherz, mv_fisherz


class TestCIT_mv_fisherz(unittest.TestCase):
    def test_chain(self):
        mv_pvalues_t1 = []
        pvalues_t1 = []
        mv_pvalues_t2 = []
        pvalues_t2 = []

        for _ in range(100):
            sz = 1000
            data = np.zeros((sz, 3))

            X = np.random.normal(0, 1.0, size=sz)
            Z = 2 * X + 0.5 * np.random.normal(0, 1.0, size=sz)
            Y = 0.5 * Z + 0.5 * np.random.normal(0, 1.0, size=sz)        
            data[:, 0], data[:, 1], data[:, 2] = X, Y, Z
            mdata = data.copy()

            # X--> Z -->Y   
            # Z -->R_Y   

            mdata[Z > 0, 1] = np.nan

            mv_pvalues_t1.append(mv_fisherz(mdata, 0, 1, ()))
            pvalues_t1.append(fisherz(data, 0, 1, ()))
            mv_pvalues_t2.append(mv_fisherz(mdata, 0, 1, (2,)))
            pvalues_t2.append(fisherz(data, 0, 1, (2,)))

        print('mv_fisherz: X and Y are not independent, pvalue is mean {:.3f}'.format(np.mean(mv_pvalues_t1)) + ' std: {:.3f}'.format(np.std(mv_pvalues_t1)))
        print('fisherz: X and Y are not independent, pvalue is mean {:.3f}'.format(np.mean(pvalues_t1)) + ' std: {:.3f}'.format(np.std(pvalues_t1)))


        print('mv_fisherz: X and Y are independent conditioning on Z, pvalue is mean {:.3f}'.format(np.mean(mv_pvalues_t2)) + ' std: {:.3f}'.format(np.std(mv_pvalues_t2)))
        print('fisherz: X and Y are independent conditioning on Z, pvalue is mean {:.3f}'.format(np.mean(pvalues_t2)) + ' std: {:.3f}'.format(np.std(pvalues_t2)))

    def test_confounder(self):
        mv_pvalues_t1 = []
        pvalues_t1 = []
        mv_pvalues_t2 = []
        pvalues_t2 = []

        for _ in range(100):
            sz = 1000
            data = np.zeros((sz, 3))

            Z = np.random.normal(0, 1.0, size=sz)
            X = 2 * Z + 0.5 * np.random.normal(0, 1.0, size=sz)
            Y = 0.5 * Z + 0.5 * np.random.normal(0, 1.0, size=sz)        
            data[:, 0], data[:, 1], data[:, 2] = X, Y, Z
            mdata = data.copy()

            # X <-- Z -->Y   
            # Z --> R_Y   

            mdata[Z > 0, 1] = np.nan

            mv_pvalues_t1.append(mv_fisherz(mdata, 0, 1, ()))
            pvalues_t1.append(fisherz(data, 0, 1, ()))
            mv_pvalues_t2.append(mv_fisherz(mdata, 0, 1, (2,)))
            pvalues_t2.append(fisherz(data, 0, 1, (2,)))

        print('mv_fisherz: X and Y are not independent, pvalue is mean {:.3f}'.format(np.mean(mv_pvalues_t1)) + ' std: {:.3f}'.format(np.std(mv_pvalues_t1)))
        print('fisherz: X and Y are not independent, pvalue is mean {:.3f}'.format(np.mean(pvalues_t1)) + ' std: {:.3f}'.format(np.std(pvalues_t1)))


        print('mv_fisherz: X and Y are independent conditioning on Z, pvalue is mean {:.3f}'.format(np.mean(mv_pvalues_t2)) + ' std: {:.3f}'.format(np.std(mv_pvalues_t2)))
        print('fisherz: X and Y are independent conditioning on Z, pvalue is mean {:.3f}'.format(np.mean(pvalues_t2)) + ' std: {:.3f}'.format(np.std(pvalues_t2)))

    def test_fork(self):
        mv_pvalues_t1 = []
        pvalues_t1 = []
        mv_pvalues_t2 = []
        pvalues_t2 = []

        for _ in range(100):
            sz = 1000
            data = np.zeros((sz, 3))

            X = np.random.normal(0, 1.0, size=sz)
            Y = np.random.normal(0, 1.0, size=sz)
            Z = 0.5 * X + 0.5 * Y + 0.5 * np.random.normal(0, 1.0, size=sz)        
            data[:, 0], data[:, 1], data[:, 2] = X, Y, Z
            mdata = data.copy()

            # X--> Z <--Y   
            # Z --> R_Y   

            mdata[Z > 0, 1] = np.nan

            mv_pvalues_t1.append(mv_fisherz(mdata, 0, 1, ()))
            pvalues_t1.append(fisherz(data, 0, 1, ()))
            mv_pvalues_t2.append(mv_fisherz(mdata, 0, 1, (2,)))
            pvalues_t2.append(fisherz(data, 0, 1, (2,)))

        print('mv_fisherz: X and Y are independent, pvalue is mean {:.3f}'.format(np.mean(mv_pvalues_t1)) + ' std: {:.3f}'.format(np.std(mv_pvalues_t1)))
        print('fisherz: X and Y are independent, pvalue is mean {:.3f}'.format(np.mean(pvalues_t1)) + ' std: {:.3f}'.format(np.std(pvalues_t1)))

        print('mv_fisherz: X and Y are not independent conditioning on Z, pvalue is mean {:.3f}'.format(np.mean(mv_pvalues_t2)) + ' std: {:.3f}'.format(np.std(mv_pvalues_t2)))
        print('fisherz: X and Y are not independent conditioning on Z, pvalue is mean {:.3f}'.format(np.mean(pvalues_t2)) + ' std: {:.3f}'.format(np.std(pvalues_t2)))

    def test_fork2(self):
        mv_pvalues_t1 = []
        pvalues_t1 = []
        mv_pvalues_t2 = []
        pvalues_t2 = []

        for _ in range(100):
            sz = 1000
            data = np.zeros((sz, 3))

            X = np.random.normal(0, 1.0, size=sz)
            Y = np.random.normal(0, 1.0, size=sz)
            Z = 0.5 * X + 0.5 * Y + 0.5 * np.random.normal(0, 1.0, size=sz)        
            data[:, 0], data[:, 1], data[:, 2] = X, Y, Z
            mdata = data.copy()

            # X--> Z <--Y   
            # Z --> R_Y   

            mdata[Y > 0, 2] = np.nan

            mv_pvalues_t1.append(mv_fisherz(mdata, 0, 1, ()))
            pvalues_t1.append(fisherz(data, 0, 1, ()))
            mv_pvalues_t2.append(mv_fisherz(mdata, 0, 1, (2,)))
            pvalues_t2.append(fisherz(data, 0, 1, (2,)))

        print('mv_fisherz: X and Y are independent, pvalue is mean {:.3f}'.format(np.mean(mv_pvalues_t1)) + ' std: {:.3f}'.format(np.std(mv_pvalues_t1)))
        print('fisherz: X and Y are independent, pvalue is mean {:.3f}'.format(np.mean(pvalues_t1)) + ' std: {:.3f}'.format(np.std(pvalues_t1)))

        print('mv_fisherz: X and Y are not independent conditioning on Z, pvalue is mean {:.3f}'.format(np.mean(mv_pvalues_t2)) + ' std: {:.3f}'.format(np.std(mv_pvalues_t2)))
        print('fisherz: X and Y are not independent conditioning on Z, pvalue is mean {:.3f}'.format(np.mean(pvalues_t2)) + ' std: {:.3f}'.format(np.std(pvalues_t2)))



        

if __name__ == '__main__':
    test = TestCIT_mv_fisherz()
    print('------------------------------')
    print('Test mv_fisherz() with the chain structure: X->Z->Y, Z -> R_Y')
    test.test_chain()
    print('------------------------------')
    print('Test mv_fisherz() with the confounder structure: X<-Z->Y, Z -> R_Y')
    test.test_confounder()
    print('------------------------------')
    print('Test mv_fisherz() with the fork structure: X->Z<-Y, Z -> R_Y')
    print('In theory, the test-wise deletion test on this graph structure leads to wrong results.')
    test.test_fork()
    print('------------------------------')
    print('Test mv_fisherz() with the fork structure: X->Z<-Y, Y -> R_Z')
    print('In theory, the test-wise deletion test on this graph structure has no problem.')
    test.test_fork2()
    