from copy import deepcopy

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.PCUtils.Helper import sort_dict_ascending


def uc_sepset(cg, priority=3, background_knowledge=None):
    '''
    Run (UC_sepset) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    '''

    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)

    R0 = []  # Records of possible orientations
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue
        if all(y not in S for S in cg.sepset[x, z]):
            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL,
                                       Endpoint.ARROW))  # Fully orient the edge irrespective of what have been oriented

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if cg_new.G.graph[x, y] == Endpoint.TAIL.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[x, y] == Endpoint.ARROW.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if cg_new.G.graph[z, y] == Endpoint.TAIL.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[z, y] == Endpoint.ARROW.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL,
                                           Endpoint.ARROW))  # Orient only if the edges have not been oriented the other way around

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            else:
                R0.append((x, y, z))

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            for (x, y, z) in R0:
                cond = cg_new.find_cond_sets_with_mid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sort_dict_ascending(UC_dict)

        else:  # 4. Order colliders by p_{xy|not y} in descending order
            for (x, y, z) in R0:
                cond = cg_new.find_cond_sets_without_mid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sort_dict_ascending(UC_dict, descending=True)

        for (x, y, z) in UC_dict.keys():
            if (background_knowledge is not None) and \
                    (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                     background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
                continue
            if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL,
                                       Endpoint.ARROW))  # Orient only if the edges have not been oriented the other way around

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

        return cg_new


def maxp(cg, priority=3, background_knowledge=None):
    '''
    Run (MaxP) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    '''

    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)

        max_p_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_with_y])
        max_p_not_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_without_y])

        if max_p_not_contain_y > max_p_contain_y:
            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL,
                                       Endpoint.ARROW))  # Fully orient the edge irrespective of what have been oriented

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is None:
                    if cg_new.G.graph[x, y] == Endpoint.TAIL.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[x, y] == Endpoint.ARROW.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is None:
                    if cg_new.G.graph[z, y] == Endpoint.TAIL.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[z, y] == Endpoint.ARROW.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL,
                                           Endpoint.ARROW))  # Orient only if the edges have not been oriented the other way around

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 3:
                UC_dict[(x, y, z)] = max_p_contain_y

            elif priority == 4:
                UC_dict[(x, y, z)] = max_p_not_contain_y

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sort_dict_ascending(UC_dict)
        else:  # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sort_dict_ascending(UC_dict, True)

        for (x, y, z) in UC_dict.keys():
            if (background_knowledge is not None) and \
                    (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                     background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
                continue

            if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL,
                                       Endpoint.ARROW))  # Orient only if the edges have not been oriented the other way around

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

        return cg_new


def definite_maxp(cg, alpha, priority=4, background_knowledge=None):
    '''
    Run (Definite_MaxP) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    '''

    assert 1 > alpha >= 0
    assert priority in [2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
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
                    break  # ambiguous triple
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

    if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
        UC_dict = sort_dict_ascending(UC_dict)
    elif priority == 4:  # 4. Order colliders by p_{xz|not y} in descending order
        UC_dict = sort_dict_ascending(UC_dict, True)

    for (x, y, z) in UC_dict.keys():
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                cg_new.G.remove_edge(edge1)
            cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL,
                                   Endpoint.ARROW))  # Orient only if the edges have not been oriented the other way around

            edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
            if edge2 is not None:
                cg_new.G.remove_edge(edge2)
            cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            cg_new.definite_UC.append((x, y, z))

    return cg_new
