'''
    File name: merge_cluster.py
    Discription: Merge overlaping cluster
    Author: ZhiyiHuang@DMIRLab, RuichuCai@DMIRLab
    Form DMIRLab: https://dmir.gdut.edu.cn/
'''

from collections import deque


def _get_all_elements(S):
    result = set()
    for i in S:
        for j in i:
            result |= {j}
    return result


def merge_overlaping_cluster(cluster_list):
    v_labels = _get_all_elements(cluster_list)
    cluster_dict = {i: -1 for i in v_labels}
    cluster_b = {i: [] for i in v_labels}
    cluster_len = 0
    for i in range(len(cluster_list)):
        for j in cluster_list[i]:
            cluster_b[j].append(i)

    visited = [False] * len(cluster_list)
    cont = True
    while cont:
        cont = False
        q = deque()
        for i, val in enumerate(visited):
            if not val:
                q.append(i)
                visited[i] = True
                break
        while q:
            top = q.popleft()
            for i in cluster_list[top]:
                cluster_dict[i] = cluster_len
                for j in cluster_b[i]:
                    if not visited[j]:
                        q.append(j)
                        visited[j] = True

        for i in visited:
            if not i:
                cont = True
                break
        cluster_len += 1

    cluster = [[] for _ in range(cluster_len)]
    for i in v_labels:
        cluster[cluster_dict[i]].append(i)

    return cluster
