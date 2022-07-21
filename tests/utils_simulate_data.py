#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

def simulate_discrete_data(
        num_of_nodes,
        sample_size,
        truth_DAG_directed_edges,
        random_seed=None):
    from pgmpy.models.BayesianNetwork import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.sampling import BayesianModelSampling

    def _simulate_cards():
        '''
        why we need this: to calculate cpd of a node with k parents,
            the conditions to be enumerated is the production of these k parents' cardinalities
            which will be exponentially slow w.r.t. k.
            so we want that, if a node has many parents (large k), these parents' cardinalities should be small

        denote peers_num: peers_num[i, j] = k (where k>0),
            means that there are k parents pointing to node i, and j is among these k parents.
        max_peers = peers_num.max(axis=0): the larger max_peers[j], the smaller card[j] should be.
        '''
        MAX_ENUMERATION_COMBINATION_NUM = 20
        in_degrees = adjacency_matrix.sum(axis=1)
        peers_num = in_degrees[:, None] * adjacency_matrix
        max_peers_num = peers_num.max(axis=0)
        max_peers_num[max_peers_num == 0] = 1 # to avoid division by 0 (for leaf nodes)
        cards = [np.random.randint(2, 1 + max(2, MAX_ENUMERATION_COMBINATION_NUM ** (1. / mpn)))
                    for mpn in max_peers_num]
        return cards

    def _random_alpha():
        DIRICHLET_ALPHA_LOWER, DIRICHLET_ALPHA_UPPER = 1., 5.
        return np.random.uniform(DIRICHLET_ALPHA_LOWER, DIRICHLET_ALPHA_UPPER)

    if random_seed is not None:
        state = np.random.get_state() # save the current random state
        np.random.seed(random_seed)  # set the random state to 42 temporarily, just for the following lines
    adjacency_matrix = np.zeros((num_of_nodes, num_of_nodes))
    adjacency_matrix[tuple(zip(*truth_DAG_directed_edges))] = 1
    adjacency_matrix = adjacency_matrix.T

    cards = _simulate_cards()
    bn = BayesianNetwork(truth_DAG_directed_edges)  # so isolating nodes will echo error
    for node in range(num_of_nodes):
        parents = np.where(adjacency_matrix[node])[0].tolist()
        parents_card = [cards[prt] for prt in parents]
        rand_ps = np.array([np.random.dirichlet(np.ones(cards[node]) * _random_alpha()) for _ in
                            range(int(np.prod(parents_card)))]).T.tolist()

        cpd = TabularCPD(node, cards[node], rand_ps, evidence=parents, evidence_card=parents_card)
        bn.add_cpds(cpd)
    inference = BayesianModelSampling(bn)
    df = inference.forward_sample(size=sample_size, show_progress=False)
    topo_order = list(map(int, df.columns))
    topo_index = [-1] * len(topo_order)
    for ind, node in enumerate(topo_order): topo_index[node] = ind
    data = df.to_numpy()[:, topo_index].astype(np.int64)

    if random_seed is not None: np.random.set_state(state) # restore the random state
    return data

def simulate_linear_continuous_data(
        num_of_nodes,
        sample_size,
        truth_DAG_directed_edges,
        noise_type='gaussian',  # currently: 'gaussian' or 'exponential'
        random_seed=None,
        linear_weight_minabs=0.5,
        linear_weight_maxabs=0.9,
        linear_weight_netative_prob=0.5):
    if random_seed is not None:
        state = np.random.get_state() # save the current random state
        np.random.seed(random_seed)  # set the random state to 42 temporarily, just for the following lines
    adjacency_matrix = np.zeros((num_of_nodes, num_of_nodes))
    adjacency_matrix[tuple(zip(*truth_DAG_directed_edges))] = 1
    adjacency_matrix = adjacency_matrix.T
    weight_mask = np.random.uniform(linear_weight_minabs, linear_weight_maxabs, (num_of_nodes, num_of_nodes))
    weight_mask[np.unravel_index(np.random.choice(np.arange(weight_mask.size), replace=False,
                size=int(weight_mask.size * linear_weight_netative_prob)), weight_mask.shape)] *= -1.
    adjacency_matrix = adjacency_matrix * weight_mask
    mixing_matrix = np.linalg.inv(np.eye(num_of_nodes) - adjacency_matrix)
    if noise_type == 'gaussian':
        exogenous_noise = np.random.normal(0, 1, (num_of_nodes, sample_size))
    elif noise_type == 'exponential':
        exogenous_noise = np.random.exponential(1, (num_of_nodes, sample_size))
    else:
        raise NotImplementedError
    data = (mixing_matrix @ exogenous_noise).T  # in shape (sample_size, num_of_nodes)
    if random_seed is not None: np.random.set_state(state) # restore the random state
    return data
