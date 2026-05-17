from __future__ import annotations
from itertools import combinations


class Cover:
    """
    Class to represent a Cover of latent variables. A Cover can either be
    atomic or non-atomic.

    An atomic Cover is one where the variables cannot be split into
    disjoint set of atomic Covers. E.g. if {L1, L2} forms a Cover, it is
    not atomic if L1 and L2 individually are atomic Covers.
    Note: Only atomic Covers can be children nodes.

    A Cover can also be temporary or not. A temporary Cover is one which was
    introduced not because a rank deficiency was found but because we had to
    introduce a temporary root variable to connect the remaining variables
    when no more rank deficient sets may be found.
    """

    def __init__(self, varnames, atomic=True, temp=False, is_observed=True, is_leaf=None):
        if isinstance(varnames, str):
            self.vars = set([varnames])

        elif isinstance(varnames, list):
            self.vars = set(varnames)

        elif isinstance(varnames, set):
            self.vars = varnames
        else:
            raise ValueError(f"{varnames} is neither str, list, set.")

       # if len(self.vars) > 0:
       #     v = next(iter(self.vars))
       #     self.type = v[:1]

        self.atomic = atomic
        self.temp = temp
        self.is_observed=is_observed
        self.is_leaf=is_leaf

    def __eq__(self, other):
        if not isinstance(other, Cover):
            return NotImplemented
        return self.vars == other.vars

    # The set of variables in any minimalGroup should be unique
    def __hash__(self):
        s = "".join(sorted(list(self.vars)))
        return hash(s)

    # Union with another Cover
    def union(self, L):
        self.vars = self.vars.union(L.vars)

    def __len__(self):
        return len(self.vars)

    @property
    def isAtomic(self):
        return self.atomic

    @property
    def isTemp(self):
        return self.temp

    def takeOne(self):
        return next(iter(self.vars))

    def __str__(self):
        if len(self.vars) == 1:
            return next(iter(self.vars))
        else:
            vars = ",".join(list(self.vars))
            return "{" + vars + "}"

    def __repr__(self):
        return str(self)

    def isSubset(self, Bs: set[Cover] | Cover, strict=False):
        if isinstance(Bs, set):
            Bvars = getVars(Bs)
        elif isinstance(Bs, Cover):
            Bvars = Bs.vars
        else:
            raise ValueError("Argument must be set of Covers or Cover.")
        if strict:
            return self.vars < Bvars
        return self.vars <= Bvars

    def intersection(self, B):
        return self.vars.intersection(B.vars)


##################################
# Methods associated with Covers #
##################################


def setLength(Vs: set[Cover]):
    """
    Determine ||Vs||
    """
    assert not isinstance(Vs, str), "Cannot be string."
    return len(getVars(Vs))


def setDifference(As: set[Cover], Bs: set[Cover]):
    diff = As - Bs  # first remove any common elements
    newset = set()
    while len(diff) > 0:
        A = diff.pop()
        newset.add(A)
        for B in Bs:
            if len(A.intersection(B)) > 0:
                newset.remove(A)
                break
    return newset


def setOverlap(As: set[Cover], Bs: set[Cover]):
    if len(As.intersection(Bs)) > 0:
        return True
    return len(setIntersection(As, Bs)) > 0


def setIntersection(As: set[Cover], Bs: set[Cover]):
    Avars = getVars(As)
    Bvars = getVars(Bs)
    return Avars.intersection(Bvars)


def getVars(As: set[Cover]):
    """
    Get all variables from a set of Covers.
    """
    vars = set()
    for A in As:
        assert isinstance(A, Cover), "Argument should be a set of Covers."
        vars.update(A.vars)
    return vars

def getOrderedVarsString(As: set[Cover]|Cover):
    """
    Get all variables from a set of Covers.
    """

    if isinstance(As, Cover):
        As = {As}

    vars = set()
    for A in As:
        assert isinstance(A, Cover), "Argument should be a set of Covers."
        vars.update(A.vars)

    vars = [x for x in vars]
    vars.sort()

    return " ".join(vars)

def deduplicate(Vs: set[Cover]):
    """
    Deduplicate cases where Vs includes {L1, {L1, L3}} into just {{L1, L3}}
    """
    newVs = set()
    for Vi in Vs:
        for Vj in Vs:
            isDuplicate = False
            if Vi.vars < Vj.vars:
                isDuplicate = True
                break
        if not isDuplicate:
            newVs.add(Vi)
    return newVs


def pairwiseOverlap(Vs: set[Cover]):
    """
    For each pair A, B in Vs, check if there are any variables overlapping.
    Return True if so.
    """
    for pair in combinations(Vs, 2):
        A, B = list(pair)
        if len(A.vars.intersection(B.vars)) > 0:
            return True
    return False
