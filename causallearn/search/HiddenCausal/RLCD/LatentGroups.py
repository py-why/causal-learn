from __future__ import annotations

import pickle
import random
from collections import deque
from copy import deepcopy
from pdb import set_trace
import networkx as nx
import math
from . import misc as M
from .GraphDrawer import DotGraph
from .Cover import Cover, setLength, getVars, setDifference, setIntersection, deduplicate
from .logger import LOGGER
import numpy as np

# Class to store discovered latent groups
class LatentGroups:
    def __init__(self, X, Xns, all_nb_set, nb_set_dict, local_Adj):# X_ns is a list of potential observed non sink variables
        self.i = 1
        self.X = set([Cover(x, is_observed=True) for x in X])
        self.activeSet = set([Cover(x, is_observed=True) for x in X])
        self.ChildrenOfNonAtomicsSet=set()
        self.latentDict = {}
        self.rankDefSets = {}
        self.clusters = {}
        self.nonAtomics = []
        self.Xns = set([x for x in Xns])
        self.activeNonSinkSet = set([x for x in Xns])
        self.X_dict = {x.takeOne():x for x in self.X}

        self.X_names = X

        self.nb_set_dict = nb_set_dict
        self.all_nb_set = all_nb_set

        self.local_Adj = local_Adj
        self.x_list_for_local_Adj = X

    def update_X_dict(self):# need to update whenever X changes
        self.X_dict = {x.takeOne():x for x in self.X}

    def get_observed_cover_by_str(self, str):
        if str in self.X_dict:
            return self.X_dict[str]
        else:
            return None

    def addRankDefSet(self, Vs, k=1, used_nonsinks=[]):
        """
        Save a rankDefSet of variables Vs, for merging into clusters later.
        """
        if not k in self.rankDefSets:
            self.rankDefSets[k] = []
        self.rankDefSets[k].append([frozenset(Vs), set(used_nonsinks)])

    def determineClusters(self):
        """
        From the saved rankDefSets, merge any pair of sets with a common
        element to derive the clusters.
        """
        k = min(list(self.rankDefSets))
        clusters = self.rankDefSets.pop(k)
        clusters_um = clusters.copy()

        n = len(clusters)

        for i in range(len(clusters)):
            clusters[i].append([clusters[i][0]])

        while True:
            i = 0
            j = 1
            while j < len(clusters):
                set1 = clusters[i][0]
                set1_nonsinks = clusters[i][1]

                set2 = clusters[j][0]
                set2_nonsinks = clusters[j][1]

                # Merge overlapping sets
                if len(setIntersection(set1, set2)) >= min(len(set1), len(set2))-1 and set1_nonsinks==set2_nonsinks:
                    Vs = set1 | set2
                    clusters[i][0] = Vs
                    #clusters[i][1] = set1_nonsinks | set2_nonsinks
                    clusters[i][1] = set1_nonsinks

                    clusters[i][2] = clusters[i][2] + clusters[j][2]

                    clusters.pop(j)

                if j >= len(clusters) - 1:
                    i += 1
                    j = i + 1
                else:
                    j += 1

            if n == len(clusters):
                break
            else:
                n = len(clusters)

        self.clusters[k] = clusters

        # Other rankDefSets of higher cardinality are discarded
        self.rankDefSets = {}

    def confirmClusters(self):
        """
        Given the clusters that we have determined, add each cluster of Vs
        as children of new latent Covers.
        Returns:
            success: boolean - whether any new cluster was successfully added
        """
        k = min(list(self.clusters))
        success = False
        clusters = self.clusters.pop(k)
        for Vs, used_nonsinks, fullVs in clusters:
            current_success = self.addCluster(Vs, fullVs, k, list(used_nonsinks))

            if current_success:

                temp_X_ls = list(self.X)
                for i, x in enumerate(temp_X_ls):
                    if len(x.vars.intersection(set(used_nonsinks)))>0:
                        if x.is_leaf is None:
                            temp_X_ls[i].is_leaf = False
                            self.Xns = self.Xns - x.vars
                
                for V in Vs:
                    if len(V.vars)==1:
                        for i, x in enumerate(temp_X_ls):
                            if len(x.vars.intersection(V.vars))>0:
                                if x.is_leaf is None:
                                    temp_X_ls[i].is_leaf = True
                                    self.Xns = self.Xns - x.vars

                self.X = set(temp_X_ls)
                self.update_X_dict()

                LOGGER.info(f"Current Xns {self.Xns}")
                #LOGGER.info(f"Current X set {self.X}")

            success = success or current_success

        self.clusters = {}
        return success

    def splitfullVs(self, Vs, fullVs):
        
        local_Adj = self.local_Adj
        x_list_for_local_Adj = self.x_list_for_local_Adj
        local_Adj = np.abs(local_Adj+np.identity(local_Adj.shape[0]))

        Vmeasures = self.pickAllMeasures(Vs)

        Vs1 = set()
        Vs2 = set()

        Vs_idx_list = []
        for V in Vmeasures:
            Vs_idx_list.append(x_list_for_local_Adj.index(list(V.vars)[0]))

        for V in Vmeasures:
            idx = x_list_for_local_Adj.index(list(V.vars)[0])
            if (local_Adj[idx].T[Vs_idx_list].T==1).all():
                Vs1.add(V)
            else:
                Vs2.add(V)

        LOGGER.debug("split Vs result: Vs=%s, Vs1=%s, Vs2=%s", Vs, Vs1, Vs2)

        return Vs1, Vs2

    def addCluster(self, Vs, fullVs, k, used_nonsinks=[]):
        """
        For a discovered cluster Vs, create a new latent Cover over it.
        Returns: a boolean indicating whether the addition of a new latent
                 relationship was successful or not (i.e. contradiction)
        """

        Vs1, Vs2 = self.splitfullVs(Vs, fullVs)

        #if setLength(Vs1) < (k-len(used_nonsinks)+1):
        #    return False

        # check cycle
        for used_nonsink in used_nonsinks:
            set1 = set()
            set2 = set()
            for x in self.findParents(self.get_observed_cover_by_str(used_nonsink)):
                set1=set1|x.vars
            for x in Vs:
                set2=set2|x.vars

            if len(set1.intersection(set2))!=0:
                LOGGER.info(
                f"Rejecting {Vs} as a cluster because {used_nonsink} is a child of {Vs}")
                return False

        parents = self.findParents(Vs)
        parentsSize = setLength(parents)
        gap = k - parentsSize
        LOGGER.info(f"Trying to add to Dict {Vs} with k={k}")

        # If gap < 0, it means that there is a contradiction, i.e. the
        # current parents of Vs has higher cardinality than the actual rank of
        # testing these Vs together.
        # However, we can just ignore this set of Vs in this case, and
        # hopefully refineClusters will correct the error later on.
        if gap < 0:
            LOGGER.info(
                f"Rejecting {Vs} as a cluster because it is rank {k}"
                f" but has parents of cardinality {parentsSize}"
            )
            return False
        
        # decide elements in the new Cover -> newCover_ls
        parents_str_set = set()
        for parent in parents:
            for V in parent.vars:
                parents_str_set.add(V)
        additional_used_nonsinks = list(set(used_nonsinks) - parents_str_set)
        newCover_ls = list(parents_str_set)
        if gap<len(additional_used_nonsinks):
            #raise ValueError
            return False

        new_latent_used = False
        for j in range(gap):
            if j < len(additional_used_nonsinks):
                newCover_ls.append(f"{additional_used_nonsinks[j]}")
            else:
                newCover_ls.append(f"L{self.i}")
                self.i += 1
                new_latent_used = True

        ########### decide is_observed
        x_str_set = set()
        for x in self.X:
            x_str_set |= (x.vars)
        if set(newCover_ls).issubset(x_str_set):
            is_observed = True
        else:
            is_observed = False
        ###########

        ######################### decide isAtomic
        def check_newCover_exists(newCover_ls):
            for L in self.latentDict.keys():
                if L.vars==set(newCover_ls):
                    return True, L.isAtomic
            return False, None
        
        newCover_exists, old_isAtomic = check_newCover_exists(newCover_ls)
        if newCover_exists:
            isAtomic = old_isAtomic
        else:
            isAtomic = new_latent_used
        ############################
        newCover = set(newCover_ls)
        newCover = Cover(newCover, isAtomic, is_observed=is_observed, is_leaf=False)
        LOGGER.info(f"--- Adding {Vs} as a {k}-cluster under {newCover}, isAtomic:{isAtomic} is_observed:{is_observed}")
        LOGGER.debug("parents=%s, parentsSize=%s, k=%s", parents, parentsSize, k)

        # Remove children who belong to a subcover
        subcovers = self.findSubcovers(newCover)
        for subcover in subcovers:
            if subcover in self.latentDict:
                Vs -= self.latentDict[subcover]["children"]

        for newCoverName in newCover_ls:
            Vs -= {self.get_observed_cover_by_str(newCoverName)}

        # Deduplicate cases where Vs includes {L1, {L1, L3}} -> {{L1, L3}}
        Vs = deduplicate(Vs)

        success = self.addOrUpdateCover(L=newCover, children=Vs, fake_children=Vs2)
        return success

    def findParents(self, Vs: set[Cover] | Cover, atomic=False, non_atomic=False):
        """
        Find parents of Vs. Returns empty set if no parents found.

        Args:
            atomic: Whether to take only atomic parents.
            non_atomic: Whether to take only non-atomic parents.
        """
        assert not (
            atomic and non_atomic
        ), "Can only specify atomic or non_atomic, not both."
        parents = set()
        if isinstance(Vs, Cover):
            Vs = set([Vs])

        for parent, values in self.latentDict.items():
            if atomic and not parent.isAtomic:
                continue
            if non_atomic and parent.isAtomic:
                continue
            for V in Vs:
                if V in values["children"]:
                    parents.add(parent)
        parents = deduplicate(parents)

        if non_atomic and len(Vs) == 1:
            assert (
                len(parents) <= 1
            ), f"{next(iter(Vs))} should not have more than one non-atomic parent."
        return parents

    def findAtomicParent(self, L):
        """
        Find the atomic parent of L.

        Raises an error if more than one atomic parent is found, which should
        not be the case.
        Returns None is L is root.
        """
        Ps = self.findParents(L, atomic=True)
        assert len(Ps) <= 1, "Nodes cannot have more than 1 atomic parent."
        P = next(iter(Ps)) if len(Ps) == 1 else None
        return P

    # Check if an AtomicGroup L has observed children
    def hasObservedChildren(self, L):
        for child in self.latentDict[L]["children"]:
            if not child.isLatent:
                return True
        return False

    def updateactiveNonSinkSet(self):
        self.activeNonSinkSet=self.Xns.copy()
        #for cover in self.activeSet:
        #    if len(cover.vars)==1 and cover.takeOne() in self.Xns:
        #        self.activeNonSinkSet.add(cover.takeOne())
    
    def updateActiveSet(self, if_for_finish=False):
        """
        Refresh the activeSet after new Covers are added.
        """
        self.activeSet = set()
        self.ChildrenOfNonAtomicsSet = set()
        # Add all measures to the activeSet
        for X in self.X:
            self.activeSet.add(X)

        # Add all atomic Covers to the activeSet
        # Non-atomic covers are never added, since the atomic covers within
        # would already be in.
        for P in self.latentDict.keys():
            if if_for_finish:
                self.activeSet.add(P)
            else:# normal mode
                if P.isAtomic:
                    self.activeSet.add(P)

        # Remove variables that are children of Covers from activeSet
        for P, val in self.latentDict.items():
            if P.isAtomic:
                self.activeSet = setDifference(self.activeSet, val["children"])
            else:
                self.activeSet = setDifference(self.activeSet, val["children"])
                self.ChildrenOfNonAtomicsSet |= val["children"]

        self.activeSet = deduplicate(self.activeSet)
        LOGGER.info(f"Active Set (if_for_finish:{if_for_finish}): {self.activeSet}")
        return


    def removeCover(self, L: Cover):
        """
        Remove an atomic Cover from the latentDict and activeSet.
        activeSet will be updated at the end to include Children of the Cover.
        """
        #assert not L.is_observed, "Can only remove latent Cover."
        #assert L.isAtomic, "Can only remove atomic Cover."

        # Get the atomicSuperCover of L and remove it
        # e.g. if L=L1 and {L1, L2} is also atomic, we must remove {L1, L2}.
        L = self.findAtomicSuperCover(L)

        # Remove all subsets of L which are also AtomicGroups
        # e.g. {L1, L2} is atomic, and L1 is atomic, so remove both {L1, L2}
        # and L1 from latentDict
        subsets = self.subsets(L)
        for subset in subsets:
            self.latentDict.pop(subset)

        for k in self.latentDict.keys():
            self.latentDict[k]["subcovers"] -= subsets
            self.latentDict[k]["children"] -= subsets

        # L may be a subset of a non-atomic Cover.
        # For this case, we only need to remove the non-atomic Cover, but the
        # other atomic Covers within can remain.
        # E.g. if L=L1 and L2 is also atomic, such that {L1, L2} is non-atomic.
        #      then we only remove {L1, L2} as a Cover but allow L2 to remain.
        nonAtomics, latentDict = self.findNonAtomics(L)
        self.latentDict = latentDict

    def findNonAtomics(self, L):
        """
        Find all nonAtomics associated with L.
        """
        latentDict = {}
        nonAtomics = {}
        for Lp, value in reversed(tuple(self.latentDict.items())):
        #for Lp, value in reversed(self.latentDict.items()):
            if not Lp.isAtomic:
                if L.vars < Lp.vars:
                    nonAtomics[Lp] = value
                    continue
            latentDict[Lp] = value
        return nonAtomics, latentDict

    def dissolveNode(self, L):
        """
        Dissolve a latent cover L by:
        1. Making it root
        2. Remove it and L's parent, and add their respective children (in the
           graph where L is root) into the activeSet
        """
        assert isinstance(L, Cover), f"{L} must be Cover"
        assert (
            len(self.activeSet) == 1
        ), f"activeSet is {self.activeSet} but should only have root variable."

        P = self.findAtomicParent(L)
        if P is None:
            # If L root, make another refined node root before continuing
            for V in self.latentDict:
                if V.isAtomic and self.isRefined(V):
                    LOGGER.info(f"{L} is root, making {V} root instead..")
                    self.makeRoot(V)
                    #printGraph(self)
                    P = self.findAtomicParent(L)
                    break
        assert (
            P is not None
        ), f"Trying to refine root {L} but no other variable available to set as root."

        # If L is an atomic cover which is a subcover of another atomicCover
        # We should dissolve the larger one instead.
        L = self.findAtomicSuperCover(L)

        LOGGER.info(f"dissolveNode {L}...")

        # Remove L and parent
        #printGraph(self)
        LOGGER.info(f"Finding non-atomics for {L}..")
        self.nonAtomics.extend(self.logNonAtomics(L))
        LOGGER.info(f"Finding non-atomics for {P}..")
        self.nonAtomics.extend(self.logNonAtomics(P))
        self.makeRoot(L)
        self.removeCover(L)
        self.removeCover(P)
        self.updateActiveSet()
        #M.display(self)
        #printGraph(self)
        return True

    # Get all AtomicGroups in a non-AtomicGroup
    def getAtomicsFromGroup(self, Ls):
        assert not Ls.isMinimal(), "Ls must not be minimal"
        groups = set()
        for subcover in self.latentDict[Ls]["subcovers"]:
            if subcover.vars <= Ls.vars:
                groups.add(subcover)
        return groups

    # Make a new connection between parent and child
    def connectNodes(self, parents, children):
        for parent in parents:
            #assert not parent.isLatent, "Parent must be latent"
            self.addOrUpdateCover(parent, children)

    # Disconnect all linkages between parent and children
    def disconnectNodes(self, parents, children, bidirectional=False):
        for parent in parents:
            self.latentDict[parent]["children"] -= children
        if bidirectional:
            # Remove edges in the other direction as well
            for child in children:
                self.latentDict[child]["children"] -= parents

    # Check if a latent has already been refined
    def isRefined(self, L):
        return self.latentDict[L].get("refined", False)

    # Reduce a list of variable sets by merging them into
    # the minimal set of non-overlapping variable sets
    def mergeList(self, Vlist):
        out = []
        mergeSuccess = False

        while len(Vlist) > 0:
            first = Vlist.pop()
            newVlist = []
            for i, Vs in enumerate(Vlist):
                commonVs = Vs.intersection(first)
                commonVs = [V for V in commonVs if not self.inLatentDict(V)]
                if len(commonVs) > 0:
                    mergeSuccess = True
                    first |= Vs
                else:
                    newVlist.append(Vs)
            Vlist = newVlist
            out.append(first)

        if not mergeSuccess:
            return out
        else:
            return self.mergeList(out)

    def containsCluster(self, Vs, nonsinks: list[str]):
        """
        Test whether the set of Covers Vs contains any subset such that the
        subset contains > k elements from an existing k-cluster.
        """
        for L, values in self.latentDict.items():
            #if L.isAtomic:
            k = len(L)
            children = self.findChildren(L)

            if len(setIntersection(Vs, children)) + len(set(nonsinks).intersection(L.vars))> k:
                return True
        return False
    
    def containsonlyaCluster(self, Vs, nonsinks: list[str]):
        """
        Test whether the set of Covers Vs contains any subset such that the
        subset contains > k elements from an existing k-cluster.
        """
        for L, values in self.latentDict.items():
            #if L.isAtomic:
            k = len(L)
            children = self.findChildren(L)

            if len(setIntersection(Vs, children))+len(setIntersection(Vs, {L}))==setLength(Vs):
            # all Vs are in children or L 
            #if len(setIntersection(Vs, children))+len(setIntersection(Vs, {L}))==setLength(Vs) and len(set(nonsinks).intersection(L.vars))==len(nonsinks):
                # all Vs are in children or L and all nonsinks are in L
                return True
        return False
    
    def checkNonSinksAreAsChildren(self, Vs, nonsinks: list[str]):
        """
        Test whether the set of Covers Vs contains any subset such that the
        subset contains > k elements from an existing k-cluster.
        """
        for L, values in self.latentDict.items():
            if len(setIntersection(Vs, {L}))>0:
                children = self.findChildren(L)
                children_str_set = set()
                for ch_cover in children:
                    children_str_set|=ch_cover.vars

                if len(set(nonsinks).intersection(children_str_set))>0:
                    return True

        return False
    
    def checkAsAllAdjacent(self, As):

        local_Adj = self.local_Adj
        x_list_for_local_Adj = self.x_list_for_local_Adj

        local_Adj = np.abs(local_Adj+np.identity(local_Adj.shape[0]))

        Ameasures = self.pickAllMeasures(As)
        idxlist = []
        for A in Ameasures:
            for x in A.vars:
                idxlist.append(x_list_for_local_Adj.index(x))

        if ((local_Adj[idxlist].T[idxlist].T)==1).all():
            return True
        else:
            return False

    def MeassuredHasNonSinks(self, As, nonsinks):
        for temp1 in self.pickAllMeasures(As):
            temp2 = temp1.vars
            if len(temp2.intersection(set(nonsinks)))>0:
                return True
        return False
    
    def overlapPaCh(self, Vs: set[Cover]):
        
        # if any V1 in Vs has intersection with any parent of V2 in Vs then return True.

        for V in Vs:
            PaV = self.findParents(V)
            temp = Vs.copy()
            temp.discard(V)
            if len(setIntersection(temp, PaV)) > 0:
                return True
            
        return False

    def parentCardinality(self, Vs):
        """
        To compute the cardinality of Vs after we replace any cluster within Vs
        by their latent parents.
        Requires a recursive call as Vs may contain nested clusters.
        """
        Vs = deepcopy(Vs)
        k1 = setLength(Vs)
        for L, _ in self.latentDict.items():
            if L.isAtomic:
                k = len(L)
                children = self.findChildren(L)
                if len(setIntersection(Vs, children)) + len(getVars(Vs).intersection(L.vars))> k:
                    Vs -= children
                    Vs -= {L}
                    for str in getVars(Vs).intersection(L.vars):
                        if str in self.X_dict:
                            Vs.discard(self.X_dict[str])
                    Vs.add(L)
        k2 = setLength(Vs)
        if k2 < k1:
            return self.parentCardinality(Vs)
        else:
            return k1

    # Check if a variable V already belongs to an AtomicGroup
    def inLatentDict(self, V):
        for _, values in self.latentDict.items():
            if V in values["children"]:
                return True
        return False

    # Given child and parent, reverse their parentage direction
    # i.e. make child the parent instead
    def reverseParentage(self, child, parent):
        #assert not parent.is_observed, "Parent is not latent"
        #assert not child.is_observed, "Child is not latent"
        # print(f"Reversing parentage! Parent:{parent} Child:{child}")

        # Remove child as a child of parent
        self.latentDict[parent]["children"] -= set([child])

        # Add parent as a child of child
        self.latentDict[child]["children"].add(parent)

    # Recursive function for use in makeRoot
    def makeRootRecursive(self, Ls, G=None):

        # Make a copy of self, to modify
        if G is None:
            G = deepcopy(self)

        # Parents of L
        # Note: We are finding parents of L based on `self`, not the modified
        #       graph G that is passed around. This is so that we don't end up
        #       in an infinite loop making L -> P then P -> L forever.
        parents = set()
        for L in Ls:
            parents.update(self.findParents(L, atomic=True))

        # If no parents, L is root. Do nothing.
        if len(parents) == 0:
            return G

        # Reverse Direction to parents
        for parent in parents:
            for L in Ls:
                G.reverseParentage(L, parent)
        G = self.makeRootRecursive(parents, G)
        return G

    def makeRoot(self, L: Cover):
        """
        Re-orient latentDict such that L becomes the root node of the graph.

        Note that this procedure does not affect non-atomic Covers.
        """
        assert isinstance(L, Cover), f"{L} must be a Cover."
        G = self.makeRootRecursive(set([L]))
        self.latentDict = G.latentDict
        self.activeSet = set([L])

    # Find all Groups that are a superset of L
    # Including L itself
    def supersets(self, L):
        groups = set()
        for group in self.latentDict:
            if group.vars >= L.vars:
                groups.add(group)
        return groups

    # Find the largest superset for L
    def supersetLargest(self, L):
        k = len(L)
        largest = None
        for group in self.latentDict:
            if group.vars >= L.vars and len(group) >= k:
                largest = group
                k = len(group)
        return largest

    # Find all Groups that are a subset of L
    # Including L itself
    def subsets(self, L):
        groups = set()
        for group in self.latentDict:
            if group.vars <= L.vars:
                groups.add(group)
        return groups

    def findAncestor(self, L: Cover):
        ancestor_set = set()
        parents_set = self.findParents(L)
        for pa in parents_set:
            ancestor_set.add(pa)
            ancestor_set |= self.findAncestor(pa)

        return ancestor_set

    def findChildrenOfAllSubSets(self, Ls: set):
        """
        Recursive search for all immediate children of an atomic Cover
        """

        children = set()

        LsVars = set()
        for x in Ls:
            LsVars |= x.vars

        for key in self.latentDict.keys():
            if key.vars.issubset(LsVars):
                children = children | self.latentDict[key]["children"]
                #for subcover in self.latentDict[key]["subcovers"]:
                #    children = children | self.findChildrenOfAllSubSets(subcover)

        return children
            

    def findChildren(self, L: Cover, rigorous=True):
        """
        Recursive search for all immediate children of an atomic Cover
        """
        # everything is recorded in latentDict
        assert L is not None, "Should not look for None."
        #assert L.isAtomic, f"{L} should be an atomic Cover."
        #assert not L.isLeaf, f"{L} should be NonLeaf."
        children = set()

        if rigorous==True:
            if L in self.latentDict:
                children = children | self.latentDict[L]["children"]
                for subcover in self.latentDict[L]["subcovers"]:
                    children = children | self.findChildren(subcover, rigorous)

            return children
            
        else:
            #if L in self.latentDict:
            #    for subcover in self.latentDict[L]["subcovers"]:
            #        children = children | self.findChildren(subcover, rigorous)

            for key in self.latentDict.keys():
                if len(set.intersection(L.vars, key.vars))!=0:
                    children = children | self.latentDict[key]["children"]

            return children

    def findDescendants(self, L: Cover, rigorous=True, visited=None):
        if visited is None:
            visited = set()
        if L in visited:
            return set()
        visited.add(L)

        descendants = set()
        children = self.findChildren(L, rigorous=rigorous)
        descendants |= children

        for ch in children:
            descendants |= self.findDescendants(ch, rigorous=rigorous, visited=visited)

        return descendants

    def findMeassuredSubset(self, L: Cover):

        assert L is not None, "Should not look for None."
        MeassuredSubset=set()

        for str in L.vars:
            if str in self.X_dict:
                MeassuredSubset.add(self.X_dict[str])
        return MeassuredSubset

    def findNonAtomicChildren(self, L: Cover):
        """
        Find all children of non-Atomic Covers of which L is a subcover.
        """
        children = set()
        for cover in self.latentDict:
            if cover.isAtomic:
                continue
            if L in self.latentDict[cover]["subcovers"]:
                children.update(self.latentDict[cover]["children"])
        return children

    def isRoot(self, L: Cover):
        """
        Check if L is root in that it has no parents.
        """
        parents = self.findParents(L)
        return len(parents) == 0

    def findRandomLatentChild(self, L: Cover):
        children = self.findChildren(L)
        latents = [child for child in children if child.isLatent]
        child = random.sample(latents, k=1)[0]
        return child

    # For a given atomic cover L, find the largest atomicCover of which L is a
    # subset. If L is not a subset of any atomicCover, returns L itself.
    def findAtomicSuperCover(self, L):
        assert isinstance(L, Cover), "L must be a Cover."
        largestCover = L
        superCoverFound = False
        for Lp in self.latentDict:
            if (L.vars < Lp.vars) and Lp.isAtomic:
                largestCover = Lp
                superCoverFound = True
                break
        if superCoverFound:
            return self.findAtomicSuperCover(largestCover)
        else:
            return L

    def bypassSingleChild(self, L):
        """
        Given a latent variable L, this function removes the only child of L
        from the graph and connects L to its grandchildren
        """
        children = self.findChildren(L)
        assert len(children) == 1, "Can only perform if single child."
        C = next(iter(children))
        grandchildren = self.findChildren(C)
        self.removeCover(C)
        self.connectNodes(set([L]), grandchildren)  # Connect to grandchild

    def pickAllMeasures(self, Ls):
        visitedA, visitedNA, measures = self._pickAllMeasures(Ls)
        return measures

    def _pickAllMeasures(self, Ls, visitedA=set(), visitedNA=set(), measures=set()):
        """
        Given a set of latent Covers, get all the measured descendants.
        This includes descendants of non-atomic covers.
        """
        visitedA = visitedA.copy()
        visitedNA = visitedNA.copy()
        measures = measures.copy()
        Q = deque()  # FIFO queue for BFS
        for L in Ls:
            if L.is_observed:
                measures.add(L)
            else:
                for l_str in L.vars:
                    temp = self.get_observed_cover_by_str(l_str)
                    if temp is not None:
                        measures.add(temp)
                Q.append(L)

        # BFS amongst atomic descendants of Ls
        while len(Q) > 0:
            L = Q.popleft()
            visitedA.add(L)

            for C in self.findChildren(L):
                if C.is_observed:
                    measures.add(C)
                else:
                    for C_str in C.vars:
                        temp = self.get_observed_cover_by_str(C_str)
                        if temp is not None:
                            measures.add(temp)
                    Q.append(C)

        # Now check if the visited nodes contain non-atomic covers
        # If yes, do DFS on the children of each non-atomic cover
        for cover in self.latentDict:
            if cover.isAtomic or (cover in visitedNA):
                continue
            if cover.isSubset(visitedA):
                visitedNA.add(cover)
                Cs = set()
                for C in self.latentDict[cover]["children"]:
                    if C.is_observed:
                        measures.add(C)
                    else:
                        Cs.add(C)
                visitedA, visitedNA, measures = self._pickAllMeasures(
                    Cs, visitedA, visitedNA, measures
                )

        return visitedA, visitedNA, measures

    def saveLatentGroup(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def addOrUpdateCover(self, L: Cover, children: set[Cover] = set(), fake_children: set[Cover] = set()):
        """
        Add a new cover to latentDict with the specified children.
        If cover exists, add the specified children to it.

        Handles the logic of adding subcovers, nonAtomic cover etc.
        """
        
        subcovers = self.findSubcovers(L)
        L.atomic = not self.isNonAtomic(L)
        if L in self.latentDict:
            
            if children==self.latentDict[L]["children"] and subcovers==self.latentDict[L]["subcovers"] \
                and fake_children==self.latentDict[L]["fake_children"]:
                return False
            else:
                self.latentDict[L]["children"].update(children)
                self.latentDict[L]["subcovers"].update(subcovers)
                self.latentDict[L]["fake_children"].update(fake_children)
                return True
        else:
            self.latentDict[L] = {
                "children": children,
                "subcovers": subcovers,
                'fake_children': fake_children,
                "refined": False,
            }
            return True

        # If L is a rediscovered non-atomic, we might need to override edge(s)
        # e.g. if we re-discover P1, P2 -> C, but C -> P1, we remove the latter
        if L.atomic:
            return
        for C, v in self.latentDict.items():
            for P in subcovers:
                if (P in v["children"]) and (C in children):
                    LOGGER.info(f"Removing {P} as a child of {C}..")
                    self.latentDict[C]["children"].remove(P)

    def introduceTempRoot(self):
        """
        Add a temporary root over all active variables such that no rank
        deficiency is introduced.

        For n := len(activeSet), we can get rank deficiency of rank k only if
        n >= 2k + 2. So the minimal k to have no rank deficiency is n < 2k + 2,
        i.e. k > n/2 - 1.
        """
        assert len(self.activeSet) > 1, "activeSet should have > 1 Cover."
        LOGGER.info(f"Introducing a temporary root over {self.activeSet}..")
        n = setLength(self.activeSet)
        k = math.ceil(n / 2 - 1 + 0.1)
        self.addCluster(Vs=self.activeSet, k=k)
        self.updateActiveSet()
        assert len(self.activeSet) == 1, "Root should be a single Cover."
        tempRoot = next(iter(self.activeSet))
        tempRoot.temp = True
        self.addOrUpdateCover(tempRoot)

    def connectVariablesChain(self, Vs: set[Cover]):
        """
        Connect variables Vs in a chain structure.
        E.g. Vs={L1, L2, L3, L4}, then:
        - {L1} -> {L2}
        - {L1, L2} -> {L3}
        - {L1, L2, L3} -> {L4}
        """
        Ls = [V for V in Vs if V.isLatent]
        Xs = [X for X in Vs if not X.isLatent]
        j = 1
        while j < len(Ls):
            Ps = set(Ls[0:j])  # Set all but first cover as Parents
            C = Ls[j]
            newCover = Cover(getVars(Ps))
            LOGGER.info(f"Setting {newCover} -----> {C}")
            self.addOrUpdateCover(newCover, children=set([C]))
            j += 1

        # Set any measured variables as children of all latents
        if len(Xs) > 0:
            Ps = set(Ls)
            Cs = set(Xs)
            newCover = Cover(getVars(Ps))
            LOGGER.info(f"Setting {newCover} -----> {Cs}")
            self.addOrUpdateCover(newCover, children=Cs)

    def logNonAtomics(self, L):
        """
        Find nonAtomics associated with L and store their information, namely
        the set of measures that define each atomic cover within the nonAtomic.
        This info will be used to re-identify the nonAtomic variables later.
        """
        nonAtomics, _ = self.findNonAtomics(L)
        infos = []
        for Lp in nonAtomics:
            spouses = []
            LOGGER.info(f"Storing info for non-atomic cover {Lp}..")
            for C in self.latentDict[Lp]["subcovers"]:
                LOGGER.info(f"Storing {C}..")

                # If C is root, we need to set a random child to be root
                # before we record C's measures. Otherwise, C will have all
                # measures recorded which carries no information.
                if self.isRoot(C):
                    child = self.findRandomLatentChild(C)
                    Gp = deepcopy(self)
                    Gp.makeRoot(child)
                    LOGGER.info(f"{C} is root, setting {child} to be root.")
                    measures = Gp.pickAllMeasures(set([C]))
                else:
                    measures = self.pickAllMeasures(set([C]))
                k = len(C)
                spouses.append((k, measures))
                LOGGER.info(f"   For {C}: {measures} measures..")
            infos.append(
                {
                    "spouses": spouses,
                    "children": self.latentDict[Lp]["children"],
                }
            )
        return infos

    def reconnectNonAtomics(self):
        """
        Rediscover the nonAtomic Cover(s) found in self.nonAtomics.
        """

        def _rediscover(nonAtomics=[], retryList=[]):
            """
            Rediscover the nonAtomic Cover corresponding to the item with
            smallest cardinality of measures, and add it back to latentDict.
            """
            # Terminate when queue is empty
            if len(nonAtomics) == 0:
                return retryList

            d, nonAtomics = _findSmallestNonAtomic(nonAtomics)

            # Find the set of covers corresponding to each atomic
            spouses, children = d["spouses"], d["children"]
            failed = False
            discoveredCovers = set()
            for (k, measures) in spouses:
                U = set([next(iter(x.vars)) for x in measures])
                covers = UG.findMinimalSepSet(U, k)
                LOGGER.info(f"Finding Min Sep Set for {U}...")

                # This attempt fails if we fail to find covers for any spouse
                if len(covers) == 0:
                    failed = True
                    retryList.append(d)
                    break

                discoveredCovers.update(covers)
                LOGGER.info(f"   Found {covers} as covers over {U}...")

            # Create the nonAtomic Cover and add it to latentDict
            if not failed:
                coverVars = set()
                for cover in discoveredCovers:
                    coverVars.update(cover.vars)
                nonAtomicCover = Cover(coverVars, atomic=False)
                self.addOrUpdateCover(nonAtomicCover, children)
                LOGGER.info(
                    f"Rediscovered {nonAtomicCover} as a non-atomic"
                    f" parent of {children}.."
                )

            # Continue search with nonAtomics
            nonAtomics = _rediscover(nonAtomics, retryList)
            return retryList

        def _findSmallestNonAtomic(nonAtomics):
            """
            Find the nonAtomicCover with the smallest cardinality of measures.
            We should identify the nonAtomicCover for him first because it is
            easiest to find.
            """
            newlist = []
            lowest = 1e9
            index = None

            # Find smallest cardinality
            for i, d in enumerate(nonAtomics):
                spouses, children = d["spouses"], d["children"]
                card = 0
                for (k, measures) in spouses:
                    card += len(measures)
                if card < lowest:
                    index = i

            # Pop the smallest guy
            for i, v in enumerate(nonAtomics):
                if i != index:
                    newlist.append(v)
            return nonAtomics[index], newlist

        if len(self.nonAtomics) == 0:
            return

        LOGGER.info("Finding nonAtomic Covers...")

        # 1. Create a copy of the graph
        # 2. Fully connect the remaining variables in activeSet
        # 3. Use the UndirectedGraph for rediscovering the nonAtomics
        Gp = deepcopy(self)
        Gp.introduceTempRoot()
        Gp.updateActiveSet()
        UG = UndirectedGraph(Gp)
        self.nonAtomics = _rediscover(self.nonAtomics)

    def findAdjacentNodes(self, L: Cover):
        """
        Find adjacent atomicCovers to L.
        """
        Ns = set()
        Gp = deepcopy(self)
        Gp.makeRoot(L)
        for C in Gp.findChildren(L):
            if C.isTemp:
                Ns.update(Gp.findChildren(C))
            else:
                Ns.add(C)
        return set([V for V in Ns if V.isAtomic])

    def findSubcovers(self, L: Cover, only_atomic=False):
        """
        Find all subcovers of L in latentDict.
        """
        subcovers = set()
        temp = set(self.latentDict.keys())|self.X
        for cover in temp:
            if only_atomic and not cover.isAtomic:
                continue
            if cover.isSubset(L, strict=True):
                subcovers.add(cover)
        return subcovers

    def isNonAtomic(self, L: Cover):
        """
        Determine if L is nonAtomic, i.e. it can be subdivided into a disjoint
        set of atomic Covers.

        Assumption: no pair of atomic Covers has overlapping variables.
        """
        subcovers = self.findSubcovers(L, only_atomic=True)
        subcoverVars = getVars(subcovers)
        return subcoverVars == L.vars

    def disconnectForNonAtomicParents(self, G: LatentGroups, P: Cover):
        """
        When testing for independence at a child of nonAtomic parent P, we need
        to represent each cover within P with its own disjoint set of variables.
        Hence we need to disconnect the graph at suitable points to achieve this.

        We use BFS to visit descendants of each subcover of P. If we visit the
        same node again, we disconnect all edges to that node.

        Returns:
            a modified LatentGroups graph.
        """
        assert not P.isAtomic, f"{P} should be non atomic."
        Gp = deepcopy(G)
        subcovers = Gp.latentDict[P]["subcovers"]
        visited = set()
        commonNodes = set()

        # First pass to find commonNodes
        for subcover in Gp.latentDict[P]["subcovers"]:
            Gp.makeRoot(subcover)
            Q = deque()
            Q.append(subcover)
            while len(Q) > 0:
                L = Q.pop()
                children = Gp.findChildren(L)
                for child in children:
                    if child in subcovers | visited:
                        commonNodes.add(child)
                    else:
                        if child.isLatent:
                            Q.append(child)
                    visited.add(child)

        # Second pass to disconnect edges
        for subcover in Gp.latentDict[P]["subcovers"]:
            Gp.makeRoot(subcover)
            Q = deque()
            Q.append(subcover)
            while len(Q) > 0:
                L = Q.pop()
                children = Gp.findChildren(L)
                for child in children:
                    if child in subcovers | commonNodes:
                        Gp.disconnectNodes(set([L]), set([child]))
                    else:
                        if child.isLatent:
                            Q.append(child)
        return Gp

    def getDotGraph(self):
        """
        Parse a LatentGroups object into a DotGraph.
        """

        def addParentToGraph(dotGraph, parent, childrenSet):

            # Add edges from children to new parents
            for P in parent.vars:
                for childGroup in childrenSet:
                    for child in childGroup.vars:
                        dotGraph.addEdge(P, child, type=1)

        G = deepcopy(self)
        Xvars = G.X

        # Add X variables
        dotGraph = DotGraph()
        for X in Xvars:
            X = next(iter(X.vars))
            dotGraph.addNode(X)

        # Add nonAtomics first
        for cover in G.latentDict:
            if cover.isAtomic:
                continue
            for L in cover.vars:
                dotGraph.addNode(L, refined=False)

        # Add atomics second so that refined gets reflected correctly
        for cover in G.latentDict:
            if not cover.isAtomic:
                continue
            refined = G.latentDict[cover].get("refined", False)
            for L in cover.vars:
                dotGraph.addNode(L, refined=refined)

        # Work iteratively through the Graph Dictionary
        while len(G.latentDict) > 0:
            parent, values = G.latentDict.popitem()
            addParentToGraph(dotGraph, parent, values["children"])

        return dotGraph

    def pruneControlSet(self, G: LatentGroups, As: set[Cover], Bs: set[Cover]):
        """
        When doing tests for independence, there may exist backdoor connections
        from variables in As to variables in Bs, hence we need to remove those
        variables with backdoor from Bs to prevent messing up the test.

        Returns:
            Bs: A pruned control set.
        """
        Gp = deepcopy(G)
        toPrune = set()
        for A in As:
            visited = set()
            Q = deque()
            Q.append(A)
            while len(Q) > 0:
                V = Q.pop()
                visited.add(V)
                if V in Bs:
                    toPrune.add(V)
                if V.isLatent:
                    for C in Gp.findChildren(V) | Gp.findNonAtomicChildren(V):
                        if not C in visited:
                            Q.append(C)
        return Bs - toPrune

    def disconnectAllEdgestoCover(self, G: LatentGroups, L: Cover):
        """
        Disconnect all edges to L.

        This differs from removeCover in that if L is part of a non-atomic
        cover {L, L2}->C, we want to retain edge from L2->C but remove edge
        from L->C.

        Returns:
            Gp: A modified LatentGroups object
        """
        assert L.isAtomic, f"{L} is not atomic."
        Gp = deepcopy(G)
        for cover, v in G.latentDict.items():

            # Remove L as an atomic parent
            if cover == L:
                Gp.latentDict.pop(L)

            # Remove L as a child of any atomic/non-atomic parent
            if L in v["children"]:
                Gp.latentDict[cover]["children"].remove(L)

            # Remove L as a co-parent, but retain remaining parents
            if not cover.isAtomic:
                if L in v["subcovers"]:
                    subcovers = G.latentDict[cover]["subcovers"] - set([L])
                    newCover = Cover(getVars(subcovers))
                    Gp.addOrUpdateCover(newCover, v["children"])
                    Gp.latentDict.pop(cover)

        return Gp

    def toNetworkX(self):
        """
        Convert graph into a networkx undirected graph.
        """
        NG = nx.Graph()
        for L, v in self.latentDict.items():
            if L.isAtomic:
                for C in v["children"]:
                    NG.add_edge(L, C)
            else:
                for S in v["subcovers"]:
                    for C in v["children"]:
                        NG.add_edge(S, C)
        return NG


def pruneGraph(G: LatentGroups, Vs: set[Cover]):
    """
    Prune away all nodes in the graph G that are descendants of Vs.

    Note that this will result in Vs becoming leaf nodes in the pruned graph.

    Returns:
        Gp: A pruned graph.
    """
    Gp = deepcopy(G)
    for V in Vs:
        assert V.isAtomic, f"{V} is not atomic."

    nodesToDrop = set()
    # BFS to add nodes
    Q = deque()
    for V in Vs:
        Q.append(V)

    while len(Q) > 0:
        A = Q.popleft()
        if A not in Vs:
            nodesToDrop.add(A)
        if A.isLatent:
            for C in Gp.findChildren(A) | Gp.findNonAtomicChildren(A):
                if not C in nodesToDrop | Vs:
                    Q.append(C)

    for node in nodesToDrop:
        if node.isLatent:
            Gp.removeCover(node)
        else:
            for P in Gp.findParents(node):
                if P in Gp.latentDict:
                    Gp.latentDict[P]["children"] -= set([node])

    Gp.X = Gp.X.intersection(Vs)
    Gp.updateActiveSet()
    return Gp



def getLfromLatentGroups(G: LatentGroups, xvars: list):
    vars_set = set()
    for cover in G.latentDict.keys():
        vars_set |= cover.vars

    lvars_set = vars_set - set(xvars)

    vars_ls = xvars + list(lvars_set)
    L = np.zeros((len(vars_ls), len(vars_ls)))

    for cover, val in G.latentDict.items():

        if cover.atomic:
            fa_list = []
            for var in cover.vars:
                index_fa = vars_ls.index(var)
                fa_list.append(index_fa)

            for fa_index1 in fa_list:
                for fa_index2 in fa_list:
                    if fa_index1!=fa_index2:
                        L[fa_index1][fa_index2]=-2

        for var in cover.vars:
            index_fa = vars_ls.index(var)
            for chcover in val['children']:
                for chvar in chcover.vars:
                    index_ch = vars_ls.index(chvar)

                    if cover.atomic:
                        #if chcover in val['fake_children']:
                        #    L[index_fa][index_ch] = 1
                        #    L[index_ch][index_fa] = -1
                        #else:
                        #    L[index_fa][index_ch] = 1
                        #    L[index_ch][index_fa] = 1
                        L[index_fa][index_ch] = -1
                        L[index_ch][index_fa] = 1
                        #L[index_fa][index_ch] = 1
                        #L[index_ch][index_fa] = 1
                    else:
                        L[index_fa][index_ch] = -1
                        L[index_ch][index_fa] = 1


    return L