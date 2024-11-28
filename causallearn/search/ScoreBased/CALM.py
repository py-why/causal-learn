import numpy as np
import torch
import torch.nn as nn
from causallearn.utils.MarkovNetwork.iamb import iamb_markov_network
from causallearn.utils.CALMUtils import *
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from typing import Any, Dict
from scipy.special import expit as sigmoid

torch.set_default_dtype(torch.double)

def calm(
    X: np.ndarray,
    lambda1: float = 0.005,
    alpha: float = 0.01,
    tau: float = 0.5,
    rho_init: float = 1e-5,
    rho_mult: float = 3,
    htol: float = 1e-8,
    subproblem_iter: int = 40000,
    standardize: bool = False,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Perform the CALM (Continuous and Acyclicity-constrained L0-penalized likelihood with estimated Moral graph) algorithm.

    Parameters
    ----------
    X : numpy.ndarray
        Input dataset of shape (n, d), where n is the number of samples, 
        and d is the number of variables.
    lambda1 : float, optional
        Coefficient for the approximated L0 penalty, which encourages sparsity in the learned graph. Default is 0.005.
    alpha : float, optional
        Significance level for conditional independence tests. Default is 0.01.
    tau : float, optional
        Temperature parameter for the Gumbel-Sigmoid. Default is 0.5.
    rho_init : float, optional
        Initial value of the penalty parameter for the acyclicity constraint. Default is 1e-5.
    rho_mult : float, optional
        Multiplication factor for rho in each iteration. Default is 3.
    htol : float, optional
        Tolerance level for acyclicity constraint. Default is 1e-8.
    subproblem_iter : int, optional
        Number of iterations for subproblem optimization. Default is 40000.
    standardize : bool, optional
        Whether to standardize the input data (mean=0, variance=1). Default is False.
    device : str, optional
        The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    Record : dict
        A dictionary containing:
        -  Record['G']: learned causal graph, a DAG, where: Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicates  i --> j.
        -  Record['B_weighted']: weighted adjacency matrix of the learned causal graph.
    """

    d = X.shape[1]
    if standardize:
        mean_X = np.mean(X, axis=0, keepdims=True)
        std_X = np.std(X, axis=0, keepdims=True)
        X = (X - mean_X) / std_X
    else:
        X = X - np.mean(X, axis=0, keepdims=True) 

    # Compute the data covariance matrix
    cov_emp = np.cov(X.T, bias=True)

    # Learn the moral graph using the IAMB Markov network
    moral_mask, _ = iamb_markov_network(X, alpha=alpha)

    # Initialize and run the CalmModel
    device = torch.device(device)
    cov_emp = torch.from_numpy(cov_emp).to(device)
    moral_mask = torch.from_numpy(moral_mask).float().to(device)

    model = CalmModel(d, moral_mask, tau=tau, lambda1=lambda1).to(device)

    # Optimization loop
    rho = rho_init
    for _ in range(100): 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(subproblem_iter): 
            optimizer.zero_grad()
            loss = model.compute_loss(cov_emp, rho)
            loss.backward(retain_graph=True)
            optimizer.step()

        with torch.no_grad():
            B_logit_copy = model.B_logit.detach().clone()
            B_logit_copy[model.moral_mask == 0] = float('-inf')
            h_sigmoid = model.compute_h(torch.sigmoid(B_logit_copy / model.tau))

        rho *= rho_mult
        if h_sigmoid.item() <= htol or rho > 1e+16:
            break

    # Extract the final binary and weighted adjacency matrices
    params_est = model.get_params()
    B_bin, B_weighted = params_est['B_bin'], params_est['B']

    node_names = [("X%d" % (i + 1)) for i in range(d)]
    nodes = [GraphNode(name) for name in node_names]
    G = GeneralGraph(nodes)

    # Add edges to the GeneralGraph based on B_bin
    for i in range(d):
        for j in range(d):
            if B_bin[i, j] == 1:
                G.add_directed_edge(nodes[i], nodes[j])

    Record = {
        "G": G,                        # GeneralGraph object representing the learned causal graph, a DAG
        "B_weighted": B_weighted       # Weighted adjacency matrix of the learned graph
    }

    return Record

class CalmModel(nn.Module):
    """
    The CALM model

    Parameters
    ----------
    d : int
        Number of variables/nodes in the graph.
    moral_mask : torch.Tensor
        Binary mask representing the moral graph structure, used to restrict possible edges.
    tau : float, optional
        Temperature parameter for the Gumbel-Sigmoid sampling, controlling the sparsity approximation. Default is 0.5.
    lambda1 : float, optional
        Coefficient for the approximated L0 penalty (sparsity term). Default is 0.005.
    """
    def __init__(self, d, moral_mask, tau=0.5, lambda1=0.005):
        super(CalmModel, self).__init__()
        self.d = d
        self.moral_mask = moral_mask
        self.tau = tau
        self.lambda1 = lambda1
        self._init_params()

    def _init_params(self):
        """Initialize parameters"""
        self.B_param = nn.Parameter(
            torch.FloatTensor(self.d, self.d).uniform_(-0.001, 0.001).to(self.moral_mask.device)
        )
        self.B_logit = nn.Parameter(
            torch.zeros(self.d, self.d).to(self.moral_mask.device)
        )

    def sample_mask(self):
        """
        Samples a binary mask B_mask based on the Gumbel-Sigmoid approximation.
        Applies the moral graph mask to restrict possible edges.
        """
        B_mask = gumbel_sigmoid(self.B_logit, tau=self.tau)
        B_mask = B_mask * self.moral_mask
        return B_mask

    @torch.no_grad()
    def get_params(self):
        """
        Returns the estimated adjacency matrix B_bin (binary) and B (weighted), thresholding at 0.5.
        """
        threshold = 0.5
        B_param = self.B_param.cpu().detach().numpy()
        B_logit = self.B_logit.cpu().detach().numpy()
        B_logit[self.moral_mask.cpu().numpy() == 0] = float('-inf')
        B_bin = sigmoid(B_logit / self.tau)
        B_bin[B_bin < threshold] = 0
        B_bin[B_bin >= threshold] = 1
        B = B_bin * B_param
        params = {'B': B, 'B_bin': B_bin}
        return params

    def compute_likelihood(self, B, cov_emp):
        """
        Computes the likelihood-based objective function for non-equal noise variance (NV) assumption.
        """
        I = torch.eye(self.d, device=self.B_param.device)
        residuals = torch.diagonal((I - B).T @ cov_emp @ (I - B))
        likelihood = 0.5 * torch.sum(torch.log(residuals)) - torch.linalg.slogdet(I - B)[1]
        return likelihood

    def compute_sparsity(self, B_mask):
        """
        Computes the sparsity penalty (approximated L0 penalty) by summing the binary entries in B_mask.
        """
        return B_mask.sum()

    def compute_h(self, B_mask):
        """
        Computes the DAG constraint term, adapted from the DAG constraint formulation
        in Yu et al. (2019). 
        """
        return torch.trace(matrix_poly(B_mask, self.d, self.B_param.device)) - self.d

    def compute_loss(self, cov_emp, rho):
        """
        Combines likelihood, approximated L0 penalty (sparsity), and DAG constraint terms into the final loss function.
        """
        B_mask = self.sample_mask()
        B = B_mask * self.B_param
        likelihood = self.compute_likelihood(B, cov_emp)
        sparsity = self.lambda1 * self.compute_sparsity(B_mask)
        h = self.compute_h(B_mask)
        loss = likelihood + sparsity + 0.5 * rho * h**2
        return loss


