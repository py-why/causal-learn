import numpy as np
import pandas as pd
import causallearn.search.HiddenCausal.GIN.mutual as MI
from sklearn import metrics

#estimating mutual information by sklearn
def independent11(x1,y1):
    x=x1.copy()
    y=y1.copy()
    length=len(x)
##    x=x.reshape(length,1)
##    y=y.reshape(length,1)
    x = list(x)
    y = list(y)

    result_NMI=metrics.normalized_mutual_info_score(x, y)
    print (result_NMI)
    return result_NMI

#estimating mutual information by Non-parametric computation of entropy and mutual-information
def independent(x1,y1):
    x=x1.copy()
    y=y1.copy()
    length=len(x)
    x=x.reshape(length,1)
    y=y.reshape(length,1)

    if length >3000:
        k=15
    else:
        k=10

    mi= MI.mutual_information((x,y),k)
    return abs(mi)




