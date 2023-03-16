import unittest
from causallearn.utils.GESUtils import *
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass

class TestLocalScore(unittest.TestCase):

    np.random.seed(10)
    X = np.random.randn(300, 1)
    X_prime = np.random.randn(300, 1)
    Y = X + 0.5 * np.random.randn(300, 1)
    Z = Y + 0.5 * np.random.randn(300, 1)
    data = np.mat(np.hstack((X, X_prime, Y, Z)))

    np.random.seed(10)
    X_slash = np.random.randint(0, 10, (300, 1))
    X_prime_slash = np.random.randint(0, 10, (300, 1))
    Y_slash = X_slash + np.random.randint(0, 10, (300, 1))
    Z_slash = Y_slash + np.random.randint(0, 10, (300, 1))
    data_slash = np.mat(np.hstack((X_slash, X_prime_slash, Y_slash, Z_slash)))

    def test_local_score_marginal_multi(self):
        parameters = {'dlabel': {}}
        for i in range(self.data.shape[1]):
            parameters['dlabel'][i] = i

        localScoreClass = LocalScoreClass(data=self.data, local_score_fun=local_score_marginal_multi, parameters=parameters)

        q = localScoreClass.score(0, [0])
        p = localScoreClass.score(0, [1])
        v = localScoreClass.score(0, [2])

        assert q < v < p

    def test_local_score_CV_multi(self):
        parameters = {'kfold': 10, 'lambda': 0.01, 'dlabel': {}}  # regularization parameter
        for i in range(self.data.shape[1]):
            parameters['dlabel'][i] = i

        localScoreClass = LocalScoreClass(data=self.data, local_score_fun=local_score_cv_multi, parameters=parameters)

        q = localScoreClass.score(0, [0])
        p = localScoreClass.score(0, [1])
        v = localScoreClass.score(0, [2])

        assert q < v < p

    def test_local_score_BIC(self):
        parameters = {}
        parameters["lambda_value"] = 2

        localScoreClass = LocalScoreClass(data=self.data, local_score_fun=local_score_BIC, parameters=parameters)

        q = localScoreClass.score(0, [0])
        p = localScoreClass.score(0, [1])
        v = localScoreClass.score(0, [2])

        assert q < v < p

    def test_local_score_CV_general(self):
        parameters = {'kfold': 10,  # 10 fold cross validation
                      'lambda': 0.01}  # regularization parameter

        localScoreClass = LocalScoreClass(data=self.data, local_score_fun=local_score_cv_general, parameters=parameters)

        q = localScoreClass.score(0, [0])
        p = localScoreClass.score(0, [1])
        v = localScoreClass.score(0, [2])

        assert q < v < p

    def test_local_score_marginal_general(self):
        parameters = {}

        localScoreClass = LocalScoreClass(data=self.data, local_score_fun=local_score_marginal_general, parameters=parameters)

        q = localScoreClass.score(0, [0])
        p = localScoreClass.score(0, [1])
        v = localScoreClass.score(0, [2])

        assert q < v < p

    def test_local_score_BDeu(self):
        localScoreClass = LocalScoreClass(data=self.data_slash, local_score_fun=local_score_BDeu, parameters=None)

        q = localScoreClass.score(0, [0])
        p = localScoreClass.score(0, [1])
        v = localScoreClass.score(0, [2])

        assert q < v < p
