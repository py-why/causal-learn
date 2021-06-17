'''
    File name: simulation.py
    Discription: Simulate data with specific noise distribution
    Author: ZhiyiHuang@DMIRLab, RuichuCai@DMIRLab
    Form DMIRLab: https://dmir.gdut.edu.cn/
'''

import numpy as np


class SyntheticData(object):
    def __init__(self, sem_type='gaussian', noise_scale=1, power=None):
        """ Syntheic data generator
        Args:
            sem_type: expected distribution type
            noise_scale: scale parameter of noise distribution in linear SEM.
            power: power of noise
        """
        self.sem_type = sem_type
        self.noise_scale = noise_scale
        self.power = power

    def _simulate_noise(self, n, d):
        """
        Args:
            n: number of sample
            d: number of dimension
        Returns:
            e: noise matrix
        """
        if self.sem_type == 'gaussian':
            e = np.random.normal(0, self.noise_scale, size=n * d).reshape((n, -1))
        elif self.sem_type == 'laplace':
            e = np.random.laplace(0, self.noise_scale, size=n * d).reshape((n, -1))
        elif self.sem_type == 'uniform':
            e = np.random.uniform(-self.noise_scale, self.noise_scale, size=n * d).reshape((n, -1))
        else:
            raise ValueError('Unknown sem type')

        if self.power is not None:
            e = np.power(e, self.power)

        return e

    @staticmethod
    def to_weight_matrix(adj_matrix, w_range=(0.5, 2.0), negative_weight=True):
        """
        Args:
            adj_matrix: Adjacency Matrix
            w_range:
            negative_weight:
        Returns:
        """
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=adj_matrix.shape)
        if negative_weight:
            U[np.random.rand(*adj_matrix.shape) < 0.5] *= -1
        return adj_matrix.astype(float) * U

    # https://github.com/ignavier/notears-tensorflow/blob/master/src/data_loader/synthetic_dataset.py
    @staticmethod
    def simulate_random_dag(d=4, degree=3, graph_type='erdos-renyi'):
        """Simulate random DAG with some expected degree.
        Args:
            d: number of nodes
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)
        Returns:
            W: weighted DAG
        """
        if graph_type == 'erdos-renyi':
            prob = float(degree) / (d - 1)
            B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            B = np.zeros([d, d])
            bag = [0]
            for ii in range(1, d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == 'full':  # ignore degree, only for experimental use
            B = np.tril(np.ones([d, d]), k=-1)
        else:
            raise ValueError('Unknown graph type')
        # random permutation
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)
        return B_perm

    def simulate_data(self, W, n):
        """
        Args:
            B: Weight matrix
            n: number of sample
        Returns:
            X: Synthetic data
        """
        if W.shape[0] != W.shape[1]:
            raise ValueError('B should be a square matrix')
        I = np.eye(W.shape[0])
        A = np.linalg.pinv(I - W)
        e = self._simulate_noise(n, W.shape[0])
        X = np.matmul(e, A)
        return X


def test():
    datagen = SyntheticData(sem_type='gaussian', noise_scale=1, power=5)
    adj_matrix = np.array([
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    W = SyntheticData.to_weight_matrix(adj_matrix)
    print(datagen.simulate_data(W, 1000))


def main():
    test()


if __name__ == '__main__':
    main()
