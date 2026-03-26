from typing import Optional
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag
from typing import Union


def ges(
    X: ndarray = None,
    score_func: str = "local_score_BIC",
    maxP: Optional[float] = None,
    parameters: Optional[Dict[str, Any]] = None,
    node_names: Union[List[str], None] = None,
    cov: Optional[ndarray] = None,
    n: Optional[int] = None,
    lambda_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Perform greedy equivalence search (GES) algorithm

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
        samples and n_features is the number of features. Can be None if cov and n are provided (only for BIC scores).
    score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
                    'local_score_CV_multi', 'local_score_marginal_multi', 'local_score_BIC', 'local_score_BDeu')).
    maxP : allowed maximum number of parents when searching the graph
    parameters : when using CV likelihood,
                  parameters['kfold']: k-fold cross validation
                  parameters['lambda']: regularization parameter
                  parameters['dlabel']: for variables with multi-dimensions,
                               indicate which dimensions belong to the i-th variable.
    cov : covariance matrix (numpy ndarray), shape (n_features, n_features). If provided together with n,
          used directly instead of computing from X. Only valid for BIC-based score functions.
    n : sample size (int). Required when cov is provided.
    lambda_value : float, optional
        Penalty hyperparameter for BIC-based score functions. Controls the sparsity of the learned graph:
          - Larger lambda_value → stronger penalty → sparser graph (fewer edges)
          - Smaller lambda_value → weaker penalty → denser graph (more edges)
          - Default is 0.5, consistent with the standard BIC penalty (0.5 * log(n) per parameter).
        This parameter is only used by BIC-based scores ('local_score_BIC', 'local_score_BIC_from_cov',
        'local_score_BIC_from_cov_deterministic'). For other score functions, it is ignored.
        If both lambda_value and parameters['lambda_value'] are provided, lambda_value takes precedence.

    Returns
    -------
    Record['G']: learned causal graph, where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicates  i --> j ,
                    Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.
    Record['update1']: each update (Insert operator) in the forward step
    Record['update2']: each update (Delete operator) in the backward step
    Record['G_step1']: learned graph at each step in the forward step
    Record['G_step2']: learned graph at each step in the backward step
    Record['score']: the score of the learned graph
    """

    # Handle covariance matrix input
    if cov is not None and n is not None:
        if X is not None:
            warnings.warn("Both X and cov/n provided. Using cov and n, ignoring X.")
        X = None
    elif X is None:
        raise ValueError("Either X or (cov, n) must be provided.")

    if X is not None and X.shape[0] < X.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    # Determine number of variables
    n_features = cov.shape[0] if cov is not None else X.shape[1]

    # Inject lambda_value into parameters if provided as a top-level argument
    if lambda_value is not None:
        if parameters is None:
            parameters = {}
        parameters["lambda_value"] = lambda_value

    if (
        score_func == "local_score_CV_general"
    ):  # % k-fold negative cross validated likelihood based on regression in RKHS
        if X is None:
            raise ValueError("local_score_CV_general requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {
                "kfold": 10,  # 10 fold cross validation
                "lambda": 0.01,
            }  # regularization parameter
        if maxP is None:
            maxP = n_features
        N = n_features
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_general, parameters=parameters
        )

    elif (
        score_func == "local_score_marginal_general"
    ):  # negative marginal likelihood based on regression in RKHS
        if X is None:
            raise ValueError("local_score_marginal_general requires raw data X, not cov/n.")
        parameters = {}
        if maxP is None:
            maxP = n_features
        N = n_features
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_general, parameters=parameters
        )

    elif (
        score_func == "local_score_CV_multi"
    ):  # k-fold negative cross validated likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if X is None:
            raise ValueError("local_score_CV_multi requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {
                "kfold": 10,
                "lambda": 0.01,
                "dlabel": {},
            }  # regularization parameter
            for i in range(n_features):
                parameters["dlabel"][i] = i
        if maxP is None:
            maxP = len(parameters["dlabel"])
        N = len(parameters["dlabel"])
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_multi, parameters=parameters
        )

    elif (
        score_func == "local_score_marginal_multi"
    ):  # negative marginal likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if X is None:
            raise ValueError("local_score_marginal_multi requires raw data X, not cov/n.")
        if parameters is None:
            parameters = {"dlabel": {}}
            for i in range(n_features):
                parameters["dlabel"][i] = i
        if maxP is None:
            maxP = len(parameters["dlabel"])
        N = len(parameters["dlabel"])
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_multi, parameters=parameters
        )

    elif (
        score_func == "local_score_BIC" or score_func == "local_score_BIC_from_cov"
    ):  # Greedy equivalence search with BIC score
        if maxP is None:
            maxP = n_features
        N = n_features
        if parameters is None:
            parameters = {}
        if "lambda_value" not in parameters:
            parameters["lambda_value"] = 0.5
        if cov is not None:
            localScoreClass = LocalScoreClass(
                data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters,
                cov=cov, n=n,
            )
        else:
            localScoreClass = LocalScoreClass(
                data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
            )

    elif score_func == "local_score_BIC_from_cov_deterministic":
        # Modified BIC for deterministic relations (used by DGES)
        from causallearn.search.ScoreBased.DGES import local_score_BIC_from_cov_deterministic
        if maxP is None:
            maxP = n_features
        N = n_features
        if parameters is None:
            parameters = {}
        if "lambda_value" not in parameters:
            parameters["lambda_value"] = 0.5
        if "det_epsilon" not in parameters:
            parameters["det_epsilon"] = 0.01
        if cov is not None:
            localScoreClass = LocalScoreClass(
                data=X, local_score_fun=local_score_BIC_from_cov_deterministic, parameters=parameters,
                cov=cov, n=n,
            )
        else:
            localScoreClass = LocalScoreClass(
                data=X, local_score_fun=local_score_BIC_from_cov_deterministic, parameters=parameters
            )

    elif score_func == "local_score_BDeu":  # Greedy equivalence search with BDeu score
        if X is None:
            raise ValueError("local_score_BDeu requires raw data X, not cov/n.")
        if maxP is None:
            maxP = n_features
        N = n_features
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BDeu, parameters=None
        )

    else:
        raise Exception("Unknown function!")
    score_func = localScoreClass

    if node_names is None:
        node_names = [("X%d" % (i + 1)) for i in range(N)]
    nodes = []

    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)

    G = GeneralGraph(nodes)
    # G = np.matlib.zeros((N, N)) # initialize the graph structure
    score = score_g(X, G, score_func, parameters)  # initialize the score

    G = pdag2dag(G)
    G = dag2cpdag(G)

    ## --------------------------------------------------------------------
    ## forward greedy search
    record_local_score = {}  # dict cache: (node, tuple(sorted(parents))) -> score
    # record_local_score{trial}{j} record the local scores when Xj as a parent
    score_new = score
    count1 = 0
    update1 = []
    G_step1 = []
    score_record1 = []
    graph_record1 = []
    while True:
        count1 = count1 + 1
        score = score_new
        score_record1.append(score)
        graph_record1.append(G)
        max_chscore = -1e7
        max_desc = []

        # Precompute graph structure for all nodes (once per iteration)
        _nbrs, _adj, _pa, _semi = precompute_graph_info(G, N)

        for i in range(N):
            for j in range(N):
                if (
                    G.graph[i, j] == 0
                    and G.graph[j, i] == 0
                    and i != j
                    and len(_pa[j]) <= maxP
                ):
                    NA = _nbrs[j] & _adj[i]
                    T0 = sorted(_nbrs[j] - _adj[i])
                    sub = Combinatorial(T0)
                    S = np.zeros(len(sub))
                    for k in range(len(sub)):
                        if S[k] < 2:
                            T_set = set(sub[k])
                            NAT = NA | T_set
                            # Validity test 1: check clique of NA ∪ T
                            V1 = check_clique_fast(G, NAT)
                            if V1:
                                if not S[k]:
                                    # Validity test 2: semi-directed path check
                                    V2 = insert_vc2_fast(j, i, NAT, _semi)
                                else:
                                    V2 = 1
                                if V2:
                                    Idx = find_subset_include(sub[k], sub)
                                    S[np.where(Idx == 1)] = 1
                                    chscore, desc, record_local_score = (
                                        insert_changed_score_fast(
                                            X, i, j, sub[k],
                                            NA, _pa[j],
                                            record_local_score,
                                            score_func,
                                            parameters,
                                        )
                                    )
                                    if chscore > max_chscore:
                                        max_chscore = chscore
                                        max_desc = desc
                            else:
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2

        if len(max_desc) != 0:
            score_new = score + max_chscore
            if score_new - score <= 0:
                break
            G = insert(G, max_desc[0], max_desc[1], max_desc[2])
            update1.append([max_desc[0], max_desc[1], max_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step1.append(G)
        else:
            score_new = score
            break

    ## --------------------------------------------------------------------
    # backward greedy search
    count2 = 0
    score_new = score
    update2 = []
    G_step2 = []
    score_record2 = []
    graph_record2 = []
    while True:
        count2 = count2 + 1
        score = score_new
        score_record2.append(score)
        graph_record2.append(G)
        max_chscore = -1e7
        max_desc = []

        # Precompute graph structure for all nodes (once per iteration)
        _nbrs, _adj, _pa, _semi = precompute_graph_info(G, N)

        for i in range(N):
            for j in range(N):
                if (j in _nbrs[i]) or (i in _pa[j]):
                    # Xi - Xj (undirected) or Xi -> Xj (directed)
                    NA = _nbrs[j] & _adj[i]
                    H0 = sorted(NA)
                    sub = Combinatorial(H0)
                    S = np.ones(len(sub))
                    for k in range(len(sub)):
                        if S[k] == 1:
                            # Delete validity: check clique of NA - H
                            H_set = set(sub[k])
                            V = check_clique_fast(G, NA - H_set)
                            if V:
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2
                        else:
                            V = 1

                        if V:
                            chscore, desc, record_local_score = (
                                delete_changed_score_fast(
                                    X, i, j, sub[k],
                                    NA, _pa[j],
                                    record_local_score,
                                    score_func,
                                    parameters,
                                )
                            )
                            if chscore > max_chscore:
                                max_chscore = chscore
                                max_desc = desc

        if len(max_desc) != 0:
            score_new = score + max_chscore
            if score_new - score <= 0:
                break
            G = delete(G, max_desc[0], max_desc[1], max_desc[2])
            update2.append([max_desc[0], max_desc[1], max_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step2.append(G)
        else:
            score_new = score
            break

    Record = {
        "update1": update1,
        "update2": update2,
        "G_step1": G_step1,
        "G_step2": G_step2,
        "G": G,
        "score": score,
    }
    return Record
