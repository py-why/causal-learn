import causallearn.search.HiddenCausal.GIN.Test_GIN_Condition as GIN #GIN based on mutual information



def LearnCausalOrder(cluster, data):
    '''
    Learning causal order for Linear Non-Gaussian Latent Variable Model with Generalized Independent Noise Condition
    Parameters
    ----------
    cluster: dic
        causal cluster learned by phase I
    data : numpy ndarray
           data set
    Returns
    -------
    causal_order : list
        causal order
    '''
    Cluster = GetCluster(cluster)
    K = []  # Initize causal order
    while(len(Cluster) >0):
        root = FindRoot(Cluster, data, K)
        K.append(root)
        Cluster.remove(root)
    return K




def FindRoot(clusters, data, causal_order):
    '''
    Find the causal order by statistics of dependence
    Parameters
    ----------
    data : data set (numpy ndarray)
    cov : covariance matrix
    clusters : clusters of observed variables
    causal_order : causal order
    Returns
    -------
    root : latent root cause
    '''
    if len(clusters) == 1:
        return clusters[0]
    root = clusters[0]
    MI_lists=[]
    for i in clusters:
        MI=0
        for j in clusters:
            if i == j:
                continue

            X = [j[0]]
            Z = []
            r_1, r_2 = array_split(i, 2)
            X = X + r_1
            Z = Z + r_2

            if causal_order:
                for k in causal_order:
                    k_1, k_2 = array_split(k, 2)
                    X = X + k_1
                    Z = Z + k_2

            print('Debug as Mutual Information: ', X , Z)
            tmi = GIN.GIN_MI(X, Z, data)
            MI+=tmi

        MI_lists.append(MI)
        print('Debug as Mutual Information: (All results of MI)', MI_lists)

    mins=MI_lists.index(min(MI_lists))
    root = clusters[mins]
    return root



def GetCluster(cluster):
    Clu = []
    key = cluster.keys()
    for i in key:
        C = cluster[i]
        for j in C:
            Clu.append(j)
    return Clu

def array_split(x, k):
    x_len = len(x)
    # div_points = []
    sub_arys = []
    start = 0
    section_len = x_len // k
    extra = x_len % k
    for i in range(extra):
        sub_arys.append(x[start:start + section_len + 1])
        start = start + section_len + 1

    for i in range(k - extra):
        sub_arys.append(x[start:start + section_len])
        start = start + section_len
    return sub_arys