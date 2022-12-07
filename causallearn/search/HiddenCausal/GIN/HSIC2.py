from kerpy.GaussianKernel import GaussianKernel
from numpy import shape,savetxt,loadtxt,transpose,shape,reshape,concatenate
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
from independence_testing.HSICBlockTestObject import HSICBlockTestObject
import numpy as np
import pandas as pd

#Fast HSIC, refer to ./kerpy-master/independence_testing/ExampleHSIC.py

def test(x,y,alph=0.05):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
##
##    kernelX=GaussianKernel()
##    kernelY=GaussianKernel()

    kernelY = GaussianKernel(float(0.1))
    kernelX=GaussianKernel(float(0.1))


    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=False, kernelY_use_median=False,
                                          rff=True, num_rfx=20, num_rfy=20, num_nullsims=500)

    pvalue = myspectralobject.compute_pvalue(x, y)

    #print(pvalue)
    if pvalue >alph:
        return True
    else:
        return False
    #return pvalue

def test2(x,y,alph=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()

    num_samples = lens

    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                    kernelX_use_median=True, kernelY_use_median=True,
                                    blocksize=80, nullvarmethod='permutation')

    pvalue = myblockobject.compute_pvalue(x, y)
    #print(pvalue)
    if pvalue >alph:
        return True
    else:
        return False



def INtest(x,y,alph=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()
##    kernelY = GaussianKernel(float(0.3))
##    kernelX=GaussianKernel(float(0.3))
    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=True, kernelY_use_median=True,
                                          rff=True, num_rfx=20, num_rfy=20, num_nullsims=1000)

    pvalue = myspectralobject.compute_pvalue(x, y)

    return pvalue

def INtest2(x,y,alph=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()
    num_samples = lens

    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                    kernelX_use_median=True, kernelY_use_median=True,
                                    blocksize=200, nullvarmethod='permutation')

    pvalue = myblockobject.compute_pvalue(x, y)

    return pvalue



def main():
    x=np.random.uniform(size=10000)
    y=np.random.uniform(size=10000)+x
    test2(x,y)

if __name__ == '__main__':
    main()
