from __future__ import annotations
from copy import deepcopy
from math import factorial as fac
from math import sqrt
import numpy as np
from numpy.linalg import matrix_rank
from scipy.stats import norm
from itertools import combinations, chain, combinations_with_replacement
import pdb
import os
import glob
from .GraphDrawer import DotGraph
from .logger import LOGGER
from .Cover import Cover

def generateSubsetMinimal(vset, k=1):
    """
    Given a set of Covers, generate all minimum subsets s.t. cardinality > k.
    """

    def recursiveSearch(d, gap, currSubset=set()):
        thread = f"currSubset: {currSubset}, d: {d}, gap is {gap}"
        d = deepcopy(d)
        currSubset = deepcopy(currSubset)

        # Terminate if empty list
        if len(d) == 0:
            return set()

        # Pop one Cover with largest cardinality
        maxDim = max(d)
        v = d[maxDim].pop()
        if len(d[maxDim]) == 0:
            d.pop(maxDim)

        # Branch to consider all cases
        # Continue current search without this element
        if len(d) > 0:
            yield from recursiveSearch(d, gap, currSubset)

        # Add this group
        if not groupInLatentSet(v, currSubset):
            currSubset.add(v)
            gap -= maxDim

        # Continue search if gap not met
        if gap >= 0 and len(d) > 0:
            yield from recursiveSearch(d, gap, currSubset)

        # End of search tree
        if gap < 0:
            yield currSubset

    #if k == 0:
    #    return set()

    # Create dictionary where key is in descending dimension size
    # and v is a list of frozensets of variables
    d = {}

    ordered_list = list(vset)
    ordered_list.sort(key=lambda x: x.__hash__())

    for v in ordered_list:
        assert isinstance(v, Cover), "Should be Cover."
        n = len(v)
        d[n] = d.get(n, set()).union([v])

    # Run recursive search
    yield from recursiveSearch(d, k)


# Check if new group of latent vars exists in a current
# list of latent vars
def groupInLatentSet(V: Cover, currSubset: set):
    for group in currSubset:
        if len(V.vars.intersection(group.vars)) > 0:
            return True
    return False


# Centre the mean of data
def meanCentre(df):
    n = df.shape[0]
    return df - df.sum(axis=0) / n


# Return n choose r
def numCombinations(n, r):
    return fac(n) // fac(r) // fac(n - r)


def getAllMeasures(latentDict, subgroups):
    measures = set()

    for subgroup in subgroups:
        values = latentDict[subgroup]
        childrenP = values["children"]
        subgroupsP = values["subgroups"]

        for child in childrenP:
            if not child.isLatent():
                measures.update(childrenP)

        if len(subgroupsP) > 0:
            measures.update(getAllMeasures(latentDict, subgroupsP))

    return measures


# Given a set of Children, try the exact same set in a dictionary
def findEntry(latentDict, refChildren, subgroupMeasures):
    for group in latentDict:
        values = latentDict[group]
        children = values["children"]
        if children == refChildren:
            subMeasures = getAllMeasures(latentDict, values["subgroups"])
            if subMeasures == subgroupMeasures:
                return True
    return False


#!! TESTS

# S: Sample Covariance
# I, J: Disjoint index sets, |I| = |J|
def traceMatrixCompound(S, I, J, k):
    X = I + J  # Union of I and J
    SijInv = np.linalg.inv(S[np.ix_(X, X)])
    Inew = [X.index(i) for i in I]
    Jnew = [X.index(j) for j in J]
    Sij = SijInv[np.ix_(Inew, Jnew)]
    Sji = S[np.ix_(J, I)]
    A = Sji @ Sij

    m = A.shape[0]
    Sum = 0
    for Vs in combinations(range(m), k):
        Vs = list(Vs)
        Atemp = A[np.ix_(Vs, Vs)]
        Sum += np.linalg.det(Atemp)
    Sum = pow(-1, k) * Sum
    # print(f"traceMatrixCompound is {Sum}")
    return Sum


def determinantVariance(S, I, J, n):
    assert len(I) == len(J), "I and J must be same length"
    m = len(I)
    X = I + J
    SijDet = np.linalg.det(S[np.ix_(I, J)])
    SijijDet = np.linalg.det(S[np.ix_(X, X)])
    # print(f"SijDet is {SijDet}")

    Sum = 0
    for k in range(m):
        Sum += (
            fac(m - k) * fac(n + 2) / fac(n + 2 - k) * traceMatrixCompound(S, I, J, k)
        )
    firstTerm = (
        fac(n)
        / fac(n - m)
        * pow(SijDet, 2)
        * (fac(n + 2) / fac(n + 2 - m) - fac(n) / fac(n - m))
    )
    secondTerm = fac(n) / fac(n - m) * SijijDet * Sum
    variance = firstTerm + secondTerm

    # Heuristic (better way to handle negative variance?)
    if variance < 0:
        return 1
    else:
        return variance


def determinantMean(S, I, J, n):
    Scatter = S * n
    x = np.linalg.det(Scatter[np.ix_(I, J)])
    return x


# Returns p value
def determinantTest(S, I, J, n):
    detMean = determinantMean(S, I, J, n)
    detVar = determinantVariance(S, I, J, n)
    zStat = abs(detMean) / sqrt(detVar)
    pValue = (1 - norm.cdf(zStat)) * 2
    return pValue


# Return true if fail to reject null
def bonferroniTest(plist, alpha):
    m = len(plist)
    return not any([p < alpha / m for p in plist])


# Return true if fail to reject null
def bonferroniHolmTest(plist, alpha):
    plist = sorted(plist)
    m = len(plist)
    tests = [p < alpha / (m + 1 - k) for k, p in enumerate(plist)]
    # print(sum(tests))
    # print(len(plist))
    # print(plist[0])
    return not any(tests)


# Given data df, bootstrap sample and make new covariance
def bootStrapCovariance(data):
    n = data.shape[0]
    index = np.random.randint(low=0, high=n, size=n)
    bootstrap = data.values[index]
    cov = 1 / (n - 1) * bootstrap.T @ bootstrap
    return cov


def scombinations(elements, k):
    combs = [set(X) for X in combinations(elements, k)]
    if len(combs) == 0:
        return [set()]
    return combs


# Given two lists of sets, get the cartesian product
def cartesian(list1, list2):
    result = []
    for l1 in list1:
        for l2 in list2:
            result.append(l1.union(l2))
    return result


# Compare two rankDicts
def cmpDict(d1, d2):
    mismatches = {}
    same = True
    for key in d1:
        if d1[key] != d2[key]:
            mismatches[key] = (d1[key], d2[key])
            same = False

        if len(mismatches) > 50:
            break
    return same, mismatches


# Equivalent graphs must have equal rank on all subcovariances
# Test each combination for equivalence
def compareGraphs(g1, g2):
    if not set(g1.xvars) == set(g2.xvars):
        print(g1.xvars, g2.xvars)
        pdb.set_trace()
        raise ValueError("X variables must be the same")
    n = len(g1.xvars)
    numbers = list(range(2, n - 1))
    combns = list(combinations_with_replacement(numbers, 2))

    for i, j in combns:
        LOGGER.info(f"Testing i={i} vs j={j}...")
        Asets = list(combinations(g1.xvars, i))
        Bsets = list(combinations(g1.xvars, j))
        for A in Asets:
            for B in Bsets:
                Aset = frozenset(A)
                Bset = frozenset(B)

                if len(Aset.intersection(Bset)) > 0:
                    continue

                A = sorted(A)
                B = sorted(B)
                cov1 = g1.subcovariance(A, B)
                rk1 = matrix_rank(cov1)
                cov2 = g2.subcovariance(A, B)
                rk2 = matrix_rank(cov2)

                if rk1 != rk2:
                    return (False, (A, B, rk1, rk2))

    return (True, None)


def display(G):
    """
    Display a prettified version of the latentDict.
    """
    LOGGER.info(f"{'='*10} Printing Current LatentDict: {'='*10}")
    LOGGER.info(f"Active Set: {','.join([str(V) for V in G.activeSet])}")

    for P, v in G.latentDict.items():
        subcovers = v["subcovers"]
        Cs = v["children"]
        Ctext = ",".join([str(C) for C in Cs])
        Ptext = str(P)

        text = f"{Ptext} : {Ctext}"
        if len(subcovers) > 0:
            text += " | "
            for subcover in subcovers:
                text += f"[{str(subcover)}]"

        if v.get("refined", False):
            text += " - Refined!"
        LOGGER.info(f"   {text}")
    LOGGER.info("=" * 50)


def powerset(iterable):
    """
    Return powerset over the elements in iterable, with the first element being
    the empty set and the final element being the full set of elements.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def clearOutputFolder():
    # Clear output folder
    files = glob.glob("output/*.pkl") + glob.glob("output/*.png")
    for f in files:
        os.remove(f)


# Insert a new item k:v in the position of key in d
def insertItemToDict(d, oldkey, newkey, newvalue):
    d1 = {}
    for k, v in d.items():
        if k == oldkey:
            d1[newkey] = newvalue
        else:
            d1[k] = v
    return d1


def extractNumber(string):
    return int("".join(filter(str.isdigit, string)))


def reorderCovers(covers):
    """
    Given a set of covers, reorder them such that the lowest number is first.
    Reduces some randomness.
    """
    covers = list(covers.copy())
    cover_dict = {}
    for cover in covers:
        digits = [extractNumber(v) for v in cover.vars]
        min_digit = min(digits)
        cover_dict[min_digit] = cover
    covers = [v for k, v in sorted(cover_dict.items())]
    return covers


def displayI(I: dict):
    for k, v in I.items():
        A, B = list(k)
        condition = [x for x in v["condition"]]
        LOGGER.info(f"   {A} indep {B} | {condition}")


class Independences(dict):
    """
    Simple wrapper over dict to do conditional update of independence relns.

    Specifically, if we already found A indep B with details (e.g. setA, setB),
    we do not want to overwrite these details when performing update.
    """

    def update(self, other: Independences):
        for k, v in other.items():
            if k in self:
                if not isinstance(self[k], dict):
                    self[k] = v
                elif len(self[k].get("setA", set())) > 0:
                    continue
                else:
                    self[k] = v
            else:
                self[k] = v


class Edges(dict):
    """
    Simple wrapper over dict to do conditional update of edges.

    Specifically, if we already found an edge A -> B, we do not want to
    overwrite it with A - B.
    """

    def update(self, other: Edges):
        for k, v in other.items():
            if k in self:
                if not isinstance(self[k], dict):
                    self[k] = v
                elif self[k][2] == 1:
                    continue
                else:
                    self[k] = v
            else:
                self[k] = v

    def getDotGraph(self):
        """
        Parse an Edges object into a DotGraph.
        """

        def addParentToGraph(G, parent, childrenSet, directed=0):
            for P in parent.vars:
                for childGroup in childrenSet:
                    for child in childGroup.vars:
                        G.addEdge(P, child, type=directed)

        E = deepcopy(self)
        G = DotGraph()

        for AB in E:
            for V in list(AB):
                for v in V.vars:
                    G.addNode(v)

        while len(E) > 0:
            _, (A, B, direction) = E.popitem()
            addParentToGraph(G, A, set([B]), direction)

        return G
