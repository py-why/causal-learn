#-------------------------------------------------------------------------------
# Name:        模块1
# Purpose:
#
# Author:      YY
#
# Created:     03/03/2021
# Copyright:   (c) YY 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import math
from scipy.stats import chi2

def FisherTest(pvals,alph=0.01):
    Fisher_Stat=0
    L = len(pvals)
    for i in range(0,L):
        if pvals[i] ==0:
            TP = 1e-05
        else:
            TP = pvals[i]

        Fisher_Stat = Fisher_Stat-2*math.log(TP)


    Fisher_pval = 1-chi2.cdf(Fisher_Stat, 2*L)  #自由度2*L

    #print(Fisher_pval)

    if Fisher_pval >alph:
        return True,Fisher_pval
    else:
        return False,Fisher_pval






def main():
    pvals = [0.01,0.9]
    FisherTest(pvals,0.1)

if __name__ == '__main__':
    main()
