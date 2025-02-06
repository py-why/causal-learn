import causallearn.utils.cit as cit
import numpy as np

def iamb_markov_network(X, alpha=0.05):
    n, d = X.shape
    markov_network_raw = np.zeros((d, d))
    total_num_ci = 0
    cond_indep_test = cit.CIT(X, 'fisherz')
    # Estimate the markov blanket for each variable
    for i in range(d):
        markov_blanket, num_ci = iamb(cond_indep_test, d, i, alpha)
        total_num_ci += num_ci
        if len(markov_blanket) > 0:
            markov_network_raw[i, markov_blanket] = 1
            markov_network_raw[markov_blanket, i] = 1

    # AND rule: (i, j) is an edge in the Markov network
    # if and only if i and j are in Markov blanket of each other
    # TODO: Check if whether we should use AND rule or OR rule
    markov_network = np.logical_and(markov_network_raw, markov_network_raw.T).astype(float)
    return markov_network, total_num_ci


def iamb(cond_indep_test, d, target, alpha):
    # Modified from: https://github.com/wt-hu/pyCausalFS/blob/master/pyCausalFS/CBD/MBs/IAMB.py
    markov_blanket = []
    num_ci = 0
    # Forward circulate phase
    circulate_flag = True
    while circulate_flag:
        # if not change, forward phase of IAMB is finished.
        circulate_flag = False
        min_pval = float('inf')
        y = None
        variables = [i for i in range(d) if i != target and i not in markov_blanket]
        for x in variables:
            num_ci += 1
            pval = cond_indep_test(target, x, markov_blanket)
            # Choose maxsize of f(X:T|markov_blanket)
            if pval <= alpha:
                if pval < min_pval:
                    min_pval = pval
                    y = x

        # if not condition independence the node,appended to markov_blanket
        if y is not None:
            markov_blanket.append(y)
            circulate_flag = True

    # Backward circulate phase
    markov_blanket_temp = markov_blanket.copy()
    for x in markov_blanket_temp:
        # Exclude variable which need test p-value
        condition_Variables=[i for i in markov_blanket if i != x]
        num_ci += 1
        pval = cond_indep_test(target, x, condition_Variables)
        if pval > alpha:
            markov_blanket.remove(x)

    return list(set(markov_blanket)), num_ci