from __future__ import annotations

import numpy as np
from scipy.special import digamma


class IGCI(object):
    '''
    Python implementation of Information-Geometric Causal Inference (IGCI)
    for bivariate causal discovery.

    IGCI infers the causal direction between two continuous variables X and Y
    under the assumption of an invertible, (near-)deterministic functional mechanism
    Y = f(X) and the postulate that the density of the cause is independent of
    the mechanism's log-slope log|f'| with respect to an appropriate reference
    measure (Daniusis et al., 2010; Janzing et al., 2012).

    Under this postulate, the forward model induces a smaller geometric complexity
    cost than the backward model:
        C_{X->Y} < C_{Y->X}  =>  infer X -> Y.

    References
    ----------
    [1] Daniusis, P., Janzing, D., Mooij, J., Zscheischler, J., Steudel, B.,
        Zhang, K., & Schölkopf, B. (2010). Inferring deterministic causal
        relations. In Proceedings of the 26th Conference on Uncertainty in
        Artificial Intelligence (UAI 2010), pp. 143-150.
    [2] Janzing, D., Mooij, J., Zhang, K., Lemeire, J., Zscheischler, J.,
        Daniušis, P., Steudel, B., & Schölkopf, B. (2012). Information-geometric
        approach to inferring causal directions. Artificial Intelligence,
        182-183, 1-31.
    [3] Zhang, S., Wu, J., Li, Z., Liu, L., Liao, J., & Sun, L. (2022).
        FIGCI: flow-based information-geometric causal inference. In
        Proceedings of the Chinese Conference on Artificial Intelligence
        (CICAI 2022), LNCS 13605, pp. 520-531.
    '''

    def __init__(self, ref_measure: str = 'uniform', estimator: str = 'slope'):
        '''
        Construct the IGCI model.

        Parameters
        ----------
        ref_measure : {'uniform', 'gaussian'}, default='uniform'
            Reference measure used for normalisation:
            - 'uniform': rescales observations to [0, 1] via min-max scaling.
              Best suited for variables with compact support.
            - 'gaussian': standardises observations to zero mean and unit
              variance. Best suited for variables with unbounded support.
        estimator : {'slope', 'entropy'}, default='slope'
            Estimator for the directional cost:
            - 'slope': evaluates consecutive differences of sorted samples
              (Daniusis et al., 2010; Janzing et al., 2012 Eq. 28).
            - 'entropy': evaluates differential entropy difference
              C_{X->Y} = H(Y~) - H(X~) using 1-spacing nearest-neighbour
              estimation (Janzing et al., 2012 Eq. 29).
        '''
        ref_measure_lower = str(ref_measure).lower()
        if ref_measure_lower not in ('uniform', 'gaussian'):
            raise ValueError(
                f"Unknown ref_measure: {ref_measure!r}. Supported options: 'uniform', 'gaussian'."
            )

        estimator_lower = str(estimator).lower()
        if estimator_lower not in ('slope', 'entropy'):
            raise ValueError(
                f"Unknown estimator: {estimator!r}. Supported options: 'slope', 'entropy'."
            )

        self.ref_measure = ref_measure_lower
        self.estimator = estimator_lower

    def _validate_and_prepare(self, data: np.ndarray, name: str) -> np.ndarray:
        '''
        Validate a single 1D/column vector and return a 1D float numpy array.
        '''
        v = np.asarray(data, dtype=float)

        if v.ndim > 2 or (v.ndim == 2 and v.shape[1] != 1):
            raise ValueError(
                f"{name} must be a 1-D array or 2-D array of shape (n, 1), got shape {v.shape}."
            )
        v = v.ravel()

        if v.size < 2:
            raise ValueError(f"{name} must contain at least 2 samples, got {v.size}.")

        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contains NaN or infinite values.")

        return v

    def _normalise(self, v: np.ndarray) -> np.ndarray:
        '''
        Normalize data according to the configured reference measure.
        '''
        if self.ref_measure == 'uniform':
            lo, hi = float(np.min(v)), float(np.max(v))
            if hi <= lo:
                raise ValueError(
                    "Cannot normalise constant variable with zero range under uniform reference measure."
                )
            return (v - lo) / (hi - lo)
        elif self.ref_measure == 'gaussian':
            sd = float(np.std(v))
            if sd <= 0.0:
                raise ValueError(
                    "Cannot normalise constant variable with zero standard deviation under gaussian reference measure."
                )
            return (v - float(np.mean(v))) / sd
        else:
            raise ValueError(f"Unknown ref_measure: {self.ref_measure!r}")

    def _compute_slope_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        Slope-based estimate of C_{X->Y} (Janzing et al., 2012, Eq. 28).

        Points are sorted by x; consecutive intervals with non-positive
        differences (ties in x or y) are skipped.
        '''
        order = np.argsort(x, kind='mergesort')
        xs, ys = x[order], y[order]

        dx = np.diff(xs)
        dy = np.abs(np.diff(ys))
        mask = (dx > 0) & (dy > 0)

        if not np.any(mask):
            return 0.0

        return float(np.mean(np.log(dy[mask] / dx[mask])))

    def _compute_spacing_entropy(self, v: np.ndarray) -> float:
        '''
        1-spacing differential entropy estimate (Janzing et al., 2012, Eq. 29).
        '''
        v_sorted = np.sort(v)
        m = v_sorted.size
        if m < 2:
            return 0.0

        d = np.diff(v_sorted)
        d_pos = d[d > 0]
        if d_pos.size == 0:
            return 0.0

        return float(np.mean(np.log(d_pos)) + digamma(m) - digamma(1))

    def cause_or_effect(self, data_x: np.ndarray, data_y: np.ndarray) -> tuple[float, float]:
        '''
        Compute IGCI directional costs in both directions (X -> Y and Y -> X).

        Parameters
        ----------
        data_x : array-like of shape (n,) or (n, 1)
            First observed variable.
        data_y : array-like of shape (n,) or (n, 1)
            Second observed variable.

        Returns
        -------
        score_forward : float
            Cost C_{X->Y} for orientation X -> Y. Smaller value indicates
            a simpler forward mechanism.
        score_backward : float
            Cost C_{Y->X} for orientation Y -> X. Smaller value indicates
            a simpler backward mechanism.

        Notes
        -----
        The inferred causal direction corresponds to the smaller cost:
        - If score_forward < score_backward: infer X -> Y.
        - If score_backward < score_forward: infer Y -> X.
        - If score_forward == score_backward: undecided.
        '''
        x = self._validate_and_prepare(data_x, "data_x")
        y = self._validate_and_prepare(data_y, "data_y")

        if x.size != y.size:
            raise ValueError(
                f"data_x and data_y must have the same length, got {x.size} and {y.size}."
            )

        x_norm = self._normalise(x)
        y_norm = self._normalise(y)

        if self.estimator == 'slope':
            score_forward = self._compute_slope_cost(x_norm, y_norm)
            score_backward = self._compute_slope_cost(y_norm, x_norm)
        elif self.estimator == 'entropy':
            hx = self._compute_spacing_entropy(x_norm)
            hy = self._compute_spacing_entropy(y_norm)
            score_forward = hy - hx
            score_backward = hx - hy
        else:
            raise ValueError(f"Unknown estimator: {self.estimator!r}")

        return score_forward, score_backward

    def predict_direction(self, data_x: np.ndarray, data_y: np.ndarray) -> int:
        '''
        Infer the causal direction between X and Y.

        Parameters
        ----------
        data_x : array-like of shape (n,) or (n, 1)
            First observed variable.
        data_y : array-like of shape (n,) or (n, 1)
            Second observed variable.

        Returns
        -------
        direction : int
            - +1 if X -> Y is inferred (score_forward < score_backward).
            - -1 if Y -> X is inferred (score_backward < score_forward).
            - 0 if the direction cannot be determined (score_forward == score_backward).
        '''
        score_forward, score_backward = self.cause_or_effect(data_x, data_y)
        if score_forward < score_backward:
            return 1
        elif score_backward < score_forward:
            return -1
        return 0

    def compute_score(self, data_x: np.ndarray, data_y: np.ndarray) -> float:
        '''
        Compute the signed orientation statistic f = score_backward - score_forward.

        A positive score indicates X -> Y, while a negative score indicates Y -> X.
        The absolute value |f| reflects orientation confidence.

        Parameters
        ----------
        data_x : array-like of shape (n,) or (n, 1)
            First observed variable.
        data_y : array-like of shape (n,) or (n, 1)
            Second observed variable.

        Returns
        -------
        score : float
            Signed confidence statistic C_{Y->X} - C_{X->Y}.
        '''
        score_forward, score_backward = self.cause_or_effect(data_x, data_y)
        return float(score_backward - score_forward)


def igci(
    data_x: np.ndarray,
    data_y: np.ndarray,
    ref_measure: str = 'uniform',
    estimator: str = 'slope',
) -> tuple[float, float]:
    '''
    Convenience function for Information-Geometric Causal Inference (IGCI).

    Parameters
    ----------
    data_x : array-like of shape (n,) or (n, 1)
        First observed variable.
    data_y : array-like of shape (n,) or (n, 1)
        Second observed variable.
    ref_measure : {'uniform', 'gaussian'}, default='uniform'
        Reference measure ('uniform' or 'gaussian').
    estimator : {'slope', 'entropy'}, default='slope'
        Estimator ('slope' or 'entropy').

    Returns
    -------
    score_forward : float
        Cost C_{X->Y} (smaller value indicates X -> Y).
    score_backward : float
        Cost C_{Y->X} (smaller value indicates Y -> X).
    '''
    model = IGCI(ref_measure=ref_measure, estimator=estimator)
    return model.cause_or_effect(data_x, data_y)
