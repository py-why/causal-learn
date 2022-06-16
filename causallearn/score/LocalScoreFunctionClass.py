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
        hash_key = f"i_{str(i)}_PAi_{str(PAi)}"
        if self.score_cache.__contains__(hash_key):
            return self.score_cache[hash_key]
        else:
            res = self.local_score_fun(self.data, i, PAi, self.parameters)
            self.score_cache[hash_key] = res
            return res
