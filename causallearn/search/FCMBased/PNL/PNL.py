import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
import math

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from causallearn.utils.KCI.KCI import KCI_UInd
import torch.autograd as autograd
import matplotlib.pyplot as plt


class MLP(nn.Module):
    """
    Python implementation MLP, which is the same of G1 and G2
    Input: X (x1 or x2)
    """

    def __init__(self, n_inputs, n_outputs, n_layers=1, n_units=100):
        """ The MLP must have the first and last layers as FC.
        :param n_inputs: input dim
        :param n_outputs: output dim
        :param n_layers: layer num = n_layers + 2
        :param n_units: the dimension of hidden layers
        :param nonlinear: nonlinear function
        """
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_units = n_units

        # create layers
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(n_units, n_units))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_units, n_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class MixGaussianLayer(nn.Module):
    def __init__(self, Mix_K=3):
        super(MixGaussianLayer, self).__init__()
        self.Mix_K = Mix_K
        self.Pi = nn.Parameter(torch.randn(self.Mix_K, 1))
        self.Mu = nn.Parameter(torch.randn(self.Mix_K, 1))
        self.Var = nn.Parameter(torch.randn(self.Mix_K, 1))

    def forward(self, x):
        Constraint_Pi = F.softmax(self.Pi, 0)
        # -(x-u)**2/(2var**2)
        Middle1 = -((x.expand(len(x), self.Mix_K) - self.Mu.T.expand(len(x), self.Mix_K)).pow(2)).div(
            2 * (self.Var.T.expand(len(x), self.Mix_K)).pow(2))
        # sum Pi*Middle/var
        Middle2 = torch.exp(Middle1).mm(Constraint_Pi.div(torch.sqrt(2 * math.pi * self.Var.pow(2))))
        # log sum
        out = sum(torch.log(Middle2))

        return out


class PNL(object):
    """
    Use of constrained nonlinear ICA for distinguishing cause from effect.
    Python Version 3.7
    PURPOSE:
          To find which one of xi (i=1,2) is the cause. In particular, this
          function does
            1) preprocessing to make xi rather clear to Gaussian,
            2) learn the corresponding 'disturbance' under each assumed causal
            direction, and
            3) performs the independence tests to see if the assumed cause if
            independent from the learned disturbance.
    """

    def __init__(self, kernelX='Gaussian', kernelY='Gaussian', mix_K=3, epochs=100000):
        '''
        Construct the ANM model.

        Parameters:
        ----------
        kernelX: kernel function for hypothetical cause
        kernelY: kernel function for estimated noise
        mix_K: number of Gaussian mixtures for independent components
        epochs: training epochs.
        '''
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.mix_K = mix_K
        self.epochs = epochs

    def nica_mnd(self, X, TotalEpoch, KofMix):
        """
        Use of "Nonlinear ICA with MND for Matlab" for distinguishing cause from effect
        PURPOSE: Performing nonlinear ICA with the minimal nonlinear distortion or smoothness regularization.

        Parameters
        ----------
        X (n*T): a matrix containing multivariate observed data. Each row of the matrix X is a observed signal.

        Returns
        ---------
        Y (n*T): the separation result.
        """
        trpattern = X.T
        trpattern = trpattern - np.tile(np.mean(trpattern, axis=1).reshape(2, 1), (1, len(trpattern[0])))
        trpattern = np.dot(np.diag(1.5 / np.std(trpattern, axis=1)), trpattern)
        # --------------------------------------------------------
        x1 = torch.from_numpy(trpattern[0, :]).type(torch.FloatTensor).reshape(-1, 1)
        x2 = torch.from_numpy(trpattern[1, :]).type(torch.FloatTensor).reshape(-1, 1)
        x1.requires_grad = True
        x2.requires_grad = True
        y1 = x1

        Final_y2 = x2
        Min_loss = float('inf')

        G1 = MLP(1, 1, n_layers=1, n_units=20)
        G2 = MLP(1, 1, n_layers=1, n_units=20)
        # MixGaussian = MixGaussianLayer(Mix_K=KofMix)
        G3 = MLP(1, 1, n_layers=1, n_units=20)
        optimizer = torch.optim.Adam([
            {'params': G1.parameters()},
            {'params': G2.parameters()},
            {'params': G3.parameters()}], lr=1e-4, betas=(0.9, 0.99))

        for epoch in range(TotalEpoch):

            y2 = G2(x2) - G1(x1)
            # y2 = x2 - G1(x1)

            loss_pdf = torch.sum((y2)**2)

            jacob = autograd.grad(outputs=G2(x2), inputs=x2, grad_outputs=torch.ones(y2.shape), create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
            loss_jacob = - torch.sum(torch.log(torch.abs(jacob) + 1e-16))

            loss = loss_jacob + loss_pdf

            if loss < Min_loss:
                Min_loss = loss
                Final_y2 = y2

            if epoch % 100 == 0:
                print('---------------------------{}-th epoch-------------------------------'.format(epoch))
                print('The Total loss is {}'.format(loss))
                print('The jacob loss is {}'.format(loss_jacob))
                print('The pdf loss is {}'.format(loss_pdf))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        plt.plot(x1.detach().numpy(), G1(x1).detach().numpy(), '.')
        plt.plot(x2.detach().numpy(), G2(x2).detach().numpy(),'.')
        plt.show()
        return y1, Final_y2

    def cause_or_effect(self, data_x, data_y):
        '''
        Fit a PNL model in two directions and test the independence between the input and estimated noise

        Parameters
        ---------
        data_x: input data (nx1)
        data_y: output data (nx1)

        Returns
        ---------
        pval_forward: p value in the x->y direction
        pval_backward: p value in the y->x direction
        '''
        kci = KCI_UInd(self.kernelX, self.kernelY)

        # Now let's see if x1 -> x2 is plausible
        data = np.concatenate((data_x, data_y), axis=1)
        y1, y2 = self.nica_mnd(data, self.epochs, self.mix_K)
        print('To see if x1 -> x2...')

        y1_np = y1.detach().numpy()
        y2_np = y2.detach().numpy()

        pval_foward, _ = kci.compute_pvalue(y1_np, y2_np)

        # Now let's see if x2 -> x1 is plausible
        y1, y2 = self.nica_mnd(data[:, [1, 0]], self.epochs, self.mix_K)
        print('To see if x2 -> x1...')

        y1_np = y1.detach().numpy()
        y2_np = y2.detach().numpy()

        pval_backward, _ = kci.compute_pvalue(y1_np, y2_np)

        return pval_foward, pval_backward
