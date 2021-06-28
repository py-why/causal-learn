#######################################################################################################################
import Algorithm1, Algorithm2, Algorithm3, time
#######################################################################################################################


def pcAlgorithm(data, alpha, test_name, stable, uc_rule, uc_priority):
    """
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level (float) in (0, 1)
    :param test_name: name of the independence test being used
           - "Fisher_Z": Fisher's Z conditional independence test
           - "Chi_sq": Chi-squared conditional independence test
           - "G_sq": G-squared conditional independence test
    :param stable: run stabilized skeleton discovery if True (default = True)
    :param uc_rule: how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    :param uc_priority: rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    :return:
    cg: a CausalGraph object
    """
    start = time.time()
    cg_1, sepsets = Algorithm1.skeletonDiscovery(data, alpha, test_name, stable)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = Algorithm2.uc_sepset(cg_1,sepset, uc_priority)
        else:
            cg_2 = Algorithm2.uc_sepset(cg_1, sepset)
        cg = Algorithm3.Meek(cg_2)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = Algorithm2.maxP(cg_1, uc_priority)
        else:
            cg_2 = Algorithm2.maxP(cg_1)
        cg = Algorithm3.Meek(cg_2)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = Algorithm2.definiteMaxP(cg_1, alpha, uc_priority)
        else:
            cg_2 = Algorithm2.definiteMaxP(cg_1, alpha)
        cg_before = Algorithm3.definite_Meek(cg_2)
        cg = Algorithm3.Meek(cg_before)
    end = time.time()

    cg.PC_elapsed = end - start

    return cg

#######################################################################################################################
