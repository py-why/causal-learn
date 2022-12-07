import causallearn.search.HiddenCausal.GIN.GIN2 as GIN2 #based on hsic to find cluster
import causallearn.search.HiddenCausal.GIN.Utils_V2 as Utils #overlap merge utils
import itertools

#return the combination of list
def FindCombination(Lists,N):
    return itertools.combinations(Lists,N)


#GIN test by fast HSIC
#X and Z are list, e.g., X=['X1','X2'] Z=['X3']
#Data.type=Pandas.DataFrame, where data.columns=['x1','x2',...]
def GIN(X,Z,data,test_function,alpha=0.05):
    return GIN2.GIN(X,Z,data,test_function,alpha)
    #HSIC with fisher method
    #return GIN2.FisherGIN(X,Z,data,alpha)


#identifying causal cluster from 1-factor to n-factor
#limit=N  is to limited the n-factor we found
def FindCluser(data, test_function, alhpa=0.05,limit=0):
    indexs=list(range(data.shape[1]))
    B = indexs.copy()
    Cluster={}
    Grlen=2

    #finding causal cluster, 1-factor to n-factor, recorded by dic type, e.g., {'1':[['x1','x2']],'2':[['x4','x5','x6']}
    while len(B) >= Grlen and len(indexs) >=2*Grlen-1:
        LatentNum=Grlen-1
        Set_P=FindCombination(B,Grlen)
        print('identifying causal cluster: '+str(LatentNum)+'-factor model:')
        for P in Set_P:
            tind=indexs.copy()
            for t in P:
                tind.remove(t)  #   tind= ALLdata\P
            if GIN(list(P),tind,data,test_function,alhpa):
                key=Cluster.keys()
                key = list(key)
                if (LatentNum) in key:
                    temp =Cluster[LatentNum]
                    temp.append(list(P))
                    Cluster[LatentNum]=temp
                else:
                    Cluster[LatentNum]=[list(P)]
        key=Cluster.keys()
        if LatentNum in key:
            Tclu=Cluster[LatentNum]
            Tclu=Utils.merge_list(Tclu)
            Cluster[LatentNum]=Tclu
            #update the B
            for i in Tclu:
                for j in i:
                    if j in B:
                        B.remove(j)


        Grlen+=1
        print('The identified cluster in identifying '+ str(LatentNum)+ '-factor model:')
        print(Cluster)

        #limit the n-factor we found
        if limit !=0 and (Grlen-1)>limit:
            break



    return Cluster



