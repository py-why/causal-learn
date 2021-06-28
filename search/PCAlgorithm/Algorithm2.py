#######################################################################################################################
from copy import deepcopy
from Helper import sortDictAscending
#######################################################################################################################


def uc_sepset(cg, sepset, priority=3):
    """
    Run (UC_sepset) to orient unshielded colliders
    :param cg: a CausalGraph object
    :param priority: rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    :return:
    cg_new: a CausalGraph object
    """
    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)

    R0 = []     # Records of possible orientations
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.findUnshieldedTriples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if all(y not in S for S in sepset[x, z]):
            if priority == 0:       # 0: overwrite
                cg_new.adjmat[y, x] = 0     # Fully orient the edge irrespective of what have been oriented
                cg_new.adjmat[x, y] = 1
                cg_new.adjmat[y, z] = 0
                cg_new.adjmat[z, y] = 1

            elif priority == 1:     # 1: orient bi-directed
                cg_new.adjmat[x, y] = 1     # Never mind the tails
                cg_new.adjmat[z, y] = 1

            elif priority == 2:     # 2: prioritize existing
                if (not cg_new.isFullyDirected(y, x)) and (not cg_new.isFullyDirected(y, z)):
                    cg_new.adjmat[x, y] = 1  # Orient only if the edges have not been oriented the other way around
                    cg_new.adjmat[z, y] = 1
            else:
                R0.append((x, y, z))

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:           # 3. Order colliders by p_{xz|y} in ascending order
            for (x, y, z) in R0:
                cond = cg_new.findCondSetsWithMid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sortDictAscending(UC_dict)

        else:                       # 4. Order colliders by p_{xy|not y} in descending order
            for (x, y, z) in R0:
                cond = cg_new.findCondSetsWithoutMid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sortDictAscending(UC_dict, descending=True)

        for (x, y, z) in UC_dict.keys():
            if (not cg_new.isFullyDirected(y, x)) and (not cg_new.isFullyDirected(y, z)):
                cg_new.adjmat[x, y] = 1     # Orient only if the edges have not been oriented the other way around
                cg_new.adjmat[z, y] = 1

        return cg_new

#######################################################################################################################


def maxP(cg, priority=3):
    """
    Run (MaxP) to orient unshielded colliders
    :param cg: a CausalGraph object
    :param priority: rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    :return:
    cg_new: a CausalGraph object
    """
    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.findUnshieldedTriples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        cond_with_y = cg_new.findCondSetsWithMid(x, z, y)
        cond_without_y = cg_new.findCondSetsWithoutMid(x, z, y)

        max_p_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_with_y])
        max_p_not_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_without_y])

        if max_p_not_contain_y > max_p_contain_y:
            if priority == 0:    # 0: overwrite
                cg_new.adjmat[y, x] = 0  # Fully orient the edge irrespective of what have been oriented
                cg_new.adjmat[x, y] = 1
                cg_new.adjmat[y, z] = 0
                cg_new.adjmat[z, y] = 1

            elif priority == 1:  # 1: orient bi-directed
                cg_new.adjmat[x, y] = 1  # Never mind the tails
                cg_new.adjmat[z, y] = 1

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.isFullyDirected(y, x)) and (not cg_new.isFullyDirected(y, z)):
                    cg_new.adjmat[x, y] = 1  # Orient only if the edges have not been oriented the other way around
                    cg_new.adjmat[z, y] = 1

            elif priority == 3:
                UC_dict[(x, y, z)] = max_p_contain_y

            elif priority == 4:
                UC_dict[(x, y, z)] = max_p_not_contain_y

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:   # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sortDictAscending(UC_dict)
        else:               # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sortDictAscending(UC_dict, True)

        for (x, y, z) in UC_dict.keys():
            if (not cg_new.isFullyDirected(y, x)) and (not cg_new.isFullyDirected(y, z)):
                cg_new.adjmat[x, y] = 1
                cg_new.adjmat[z, y] = 1

        return cg_new

#######################################################################################################################


def definiteMaxP(cg, alpha, priority=4):
    """
    Run (Definite_MaxP) to orient unshielded colliders
    :param cg: a CausalGraph object
    :param alpha: desired significance level in (0, 1) (float)
    :param priority: rule of resolving conflicts between unshielded colliders (default = 4)
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    :return:
    cg_new: a CausalGraph object
    """
    assert 1 > alpha >= 0
    assert priority in [2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.findUnshieldedTriples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        cond_with_y = cg_new.findCondSetsWithMid(x, z, y)
        cond_without_y = cg_new.findCondSetsWithoutMid(x, z, y)
        max_p_contain_y = 0
        max_p_not_contain_y = 0
        uc_bool = True
        nuc_bool = True

        for S in cond_with_y:
            p = cg_new.ci_test(x, z, S)
            if p > alpha:
                uc_bool = False
                break
            elif p > max_p_contain_y:
                max_p_contain_y = p

        for S in cond_without_y:
            p = cg_new.ci_test(x, z, S)
            if p > alpha:
                nuc_bool = False
                if not uc_bool:
                    break       # ambiguous triple
            if p > max_p_not_contain_y:
                max_p_not_contain_y = p

        if uc_bool:
            if nuc_bool:
                if max_p_not_contain_y > max_p_contain_y:
                    if priority in [2, 3]:
                        UC_dict[(x, y, z)] = max_p_contain_y
                    if priority == 4:
                        UC_dict[(x, y, z)] = max_p_not_contain_y
                else:
                    cg_new.definite_non_UC.append((x, y, z))
            else:
                if priority in [2, 3]:
                    UC_dict[(x, y, z)] = max_p_contain_y
                if priority == 4:
                    UC_dict[(x, y, z)] = max_p_not_contain_y

        elif nuc_bool:
            cg_new.definite_non_UC.append((x, y, z))

    if priority == 3:    # 3. Order colliders by p_{xz|y} in ascending order
        UC_dict = sortDictAscending(UC_dict)
    elif priority == 4:  # 4. Order colliders by p_{xz|not y} in descending order
        UC_dict = sortDictAscending(UC_dict, True)

    for (x, y, z) in UC_dict.keys():
        if (not cg_new.isFullyDirected(y, x)) and (not cg_new.isFullyDirected(y, z)):
            cg_new.adjmat[x, y] = 1
            cg_new.adjmat[z, y] = 1
            cg_new.definite_UC.append((x, y, z))

    return cg_new

#######################################################################################################################
