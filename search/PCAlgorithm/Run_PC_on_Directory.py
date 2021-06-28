#######################################################################################################################
from datetime import datetime
from os import listdir, path
from GraphClass import tetradToCausalGraph
from PC import pcAlgorithm
import numpy as np
#######################################################################################################################


def estimateAndOutput(data_dir, output_dir, alpha, test, stable, uc_rule, uc_priority):
    """ run PC algorithm over each data in data_dir and output all the estimated graphs in output_dir
    :param data_dir: name of the directory storing the datasets (string)
    :param output_dir: name of the directory returning the estimated graphs (string)
    :param alpha: desired significance level (float) in (0, 1)
    :param test: name of the independence test being used
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
    """
    assert path.exists(data_dir)
    assert path.exists(output_dir)

    data_inputs = listdir(data_dir)
    for file in range(len(data_inputs)):
        data_file_name = str(data_inputs[file])
        data_path = data_dir + "/" + data_file_name
        data = np.loadtxt(data_path, skiprows=1)
        cg = pcAlgorithm(data, alpha, test, stable, uc_rule, uc_priority)
        print(f"[Run {file+1}] PC Elapsed time: {round(cg.PC_elapsed,3)} seconds")
        cg.rearrange(data_path)
        output_path = output_dir + "/output." + data_file_name
        cg.toTetradTxt(output_path)

#######################################################################################################################


def estimateAndCompare(data_dir, truth_dir, alpha, test, stable=True, uc_rule=0, uc_priority=-1,
                       compare_pattern=True, adj_only=False, uc_also=True, **kwargs):
    """ run PC algorithm over each data in data_dir, and compare each with the true graph in truth_dir
    :param data_dir: name of the directory storing the datasets (string)
    :param truth_dir: name of the directory storing the true graphs (string)
    :param alpha: desired significance level in (0, 1) (float)
    :param test: name of the independence test being used
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
    :param compare_pattern: compare the CausalGraph with the true pattern if True \
            and compare with the true DAG otherwise (default = True)
    :param adj_only: return only adjacency-related performance statistics if True (default = False)
    :param uc_also: return unshielded colliders-related performance statistics if True (default = True)
    :param kwargs: specify the output path of the performance statistics .TXT file if "stat_path" is in kwargs
    :return:
    overall_stat: overall performance statistics (np.ndarray)
    """
    assert path.exists(data_dir)
    assert path.exists(truth_dir)

    data_inputs = listdir(data_dir)
    stat_overall_list = []
    for file in range(len(data_inputs)):
        data_file_name = str(data_inputs[file])
        data_path = data_dir + "/" + data_file_name
        data = np.loadtxt(data_path, skiprows=1)
        cg = pcAlgorithm(data, alpha, test, stable, uc_rule, uc_priority)
        print(f"[Run {file + 1}] PC elapsed time: {round(cg.PC_elapsed, 3)} seconds")
        cg.rearrange(data_path)

        truth_path = truth_dir + "/" + str(listdir(truth_dir)[file])
        truth = tetradToCausalGraph(truth_path)
        stat_list = cg.comparison(truth, compare_pattern, adj_only, uc_also, print_to_console=False)
        stat_list.append(cg.PC_elapsed)
        stat_overall_list.append(stat_list)

    stat_overall_list = np.array(stat_overall_list)
    mean_stat = np.nanmean(stat_overall_list, axis=0)
    std_stat = np.nanstd(stat_overall_list, axis=0)
    max_stat = np.nanmax(stat_overall_list, axis=0)
    min_stat = np.nanmin(stat_overall_list, axis=0)
    overall_stat = np.round(np.transpose(np.array([mean_stat, std_stat, max_stat, min_stat])), 3)

    timestamp = str(datetime.now().strftime("%y_%m_%d_%H_%M_%S"))
    if "stat_path" in kwargs:
        stat_path = kwargs["stat_path"]
    else:
        stat_path = "results/performance_stat_" + timestamp + ".txt"

    file = open(str(stat_path), 'w')
    file.write(f'Number of runs:                                    {len(data_inputs)} \n')
    file.write(f'Significance level:                                {alpha} \n')
    file.write(f'Conditional independence test:                     {test} \n')
    file.write(f'PC stable:                                         {stable}\n')
    uc_rule_index = ["Sepset", "Max-p", "Definite Max-p"]
    file.write(f'Method of orienting unshielded colliders:          {uc_rule_index[uc_rule]} \n')
    uc_priority_index = ["overwrite", "orient bi-directed", "prioritize existing colliders",
                         "prioritize stronger colliders", "prioritize stronger* colliders"]
    if uc_priority != -1:
        file.write(f'Rule of resolving conflicts:                       {uc_priority_index[uc_priority]}\n')
    else:
        uc_priority_default_index = ["prioritize stronger colliders", "prioritize stronger colliders",
                                     "prioritize stronger* colliders"]
        file.write(f'Rule of resolving conflicts:                       {uc_priority_default_index[uc_rule]}\n')
    file.write(f'Compare to true pattern:                           {compare_pattern}\n')
    file.write('\n')

    file.write("Performance statistics                              [mean, stdev, max, min]\n")
    file.write("--------------------------------------------------------------------------------\n")
    file.write(f"Adjacency precision:                               {list(overall_stat[0, :])}\n")
    file.write(f"Adjacency recall:                                  {list(overall_stat[1, :])}\n")
    file.write(f"Adjacency F1-score:                                {list(overall_stat[2, :])}\n")
    if not adj_only:
        file.write(f"Arrowhead precision:                               {list(overall_stat[3, :])}\n")
        file.write(f"Arrowhead recall:                                  {list(overall_stat[4, :])}\n")
        file.write(f"Arrowhead F1-score:                                {list(overall_stat[5, :])}\n")
        file.write(f"Arrowhead precision (per common edges):            {list(overall_stat[6, :])}\n")
        file.write(f"Arrowhead recall (per common edges):               {list(overall_stat[7, :])}\n")
        file.write(f"Arrowhead F1-score (per common edges):             {list(overall_stat[8, :])}\n")
        if uc_also:
            file.write(f"Unshielded colliders precision:                    {list(overall_stat[9, :])}\n")
            file.write(f"Unshielded colliders recall:                       {list(overall_stat[10, :])}\n")
            file.write(f"Unshielded colliders F1-score:                     {list(overall_stat[11, :])}\n")
            file.write(f"Unshielded colliders precision (per common edges): {list(overall_stat[12, :])}\n")
            file.write(f"Unshielded colliders recall (per common edges):    {list(overall_stat[13, :])}\n")
            file.write(f"Unshielded colliders F1-score (per common edges):  {list(overall_stat[14, :])}\n")
    file.write(f"PC elapsed time (in seconds):                      {list(overall_stat[-1, :])}\n")
    file.close()

    return overall_stat

#######################################################################################################################
# estimateAndOutput("test/data", "test/output", 0.01, "Fisher_Z", True, 0, -1)
# estimateAndCompare("test/data", "test/graph", 0.01, "Fisher_Z", True, 1, -1)
#######################################################################################################################
