import causallearn.search.HiddenCausal.GIN.Test_GIN_Condition as Test_GIN_Condition #based on hsic to find cluster
import causallearn.search.HiddenCausal.GIN.Merge_Cluster as Merge_Cluster #overlap merge utils
import itertools


def FindCluser(data, test_function, alpha=0.05, Max_Factor = 2):
    '''
    Learning causal Cluster for Linear Non-Gaussian Latent Variable Model with Generalized Independent Noise Condition
    Parameters
    ----------
    data : numpy ndarray
           data set
    indep_test_method : str, default='kci'
        the name of the independence test being used
    alpha : float, default=0.05
        desired significance level of independence tests (p_value) in (0,1)
    Max_Factor: int
        Maximum number of factors
    Returns
    -------
    Cluster: dic
        Causal Cluster, e.g., {'1':[['x1','x2']],'2':[['x4','x5','x6']}
    '''
    #Initialize variable set
    indexs = list(range(data.shape[1]))
    B = indexs.copy()
    Cluster = {}
    Grlen = 2

    while len(B) >= Grlen and len(indexs) >=2*Grlen-1:
        LatentNum = Grlen-1
        Set_P = itertools.combinations(B, Grlen)
        print('Identifying causal cluster with '+str(LatentNum)+'-factor model:')
        for P in Set_P:
            tind = indexs.copy()
            for t in P:
                tind.remove(t)  #   tind= ALLdata\P
            if GIN(list(P), tind, data, test_function, alpha):
                key = list(Cluster.keys())
                if (LatentNum) in key:
                    temp = Cluster[LatentNum]
                    temp.append(list(P))
                    Cluster[LatentNum] = temp
                else:
                    Cluster[LatentNum] = [list(P)]

        print('Debug------------',Cluster)
        # Merge overlap cluster and update dataset
        key = Cluster.keys()
        if LatentNum in key:
            Tclu = Merge_Cluster.merge_list(Cluster[LatentNum])
            Cluster[LatentNum] = Tclu
            for i in Tclu:
                for j in i:
                    if j in B:
                        B.remove(j)

        Grlen += 1
        print('The identified cluster for '+ str(LatentNum)+ '-factor model:', Cluster)

        if Max_Factor !=0 and (Grlen-1)>Max_Factor:
            break

    return Cluster



def GIN(X, Z, data, test_function, alpha=0.05):
    return Test_GIN_Condition.GIN(X, Z, data, test_function, alpha)
    #Fisher method
    #return Test_GIN_Condition.FisherGIN(X, Z, data, test_function, alpha)

