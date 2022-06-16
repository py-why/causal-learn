import math
from functools import lru_cache
from typing import Any, Callable, Dict, List

import pandas as pd
from causallearn.score.LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC,
    local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from causallearn.utils.ScoreUtils import *
from numpy import ndarray


class LocalScoreClass(object):
    def __init__(
        self,
        data: Any,
        local_score_fun: Callable[[Any, int, List[int], Any], float],
        parameters=None,
    ):
        self.data = data
        self.local_score_fun = local_score_fun
        self.parameters = parameters
        self.score_cache = {}

    def score(self, i: int, PAi: List[int]) -> float:
        if i not in self.score_cache:
            self.score_cache[i] = {}

        hash_key = tuple(sorted(PAi))

        if not self.score_cache[i].__contains__(hash_key):
            self.score_cache[i][hash_key] = self.local_score_fun(self.data, i, PAi, self.parameters)

        return self.score_cache[i][hash_key]
