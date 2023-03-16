from functools import lru_cache

import networkx as nx
import numpy as np
from matplotlib import pyplot, transforms
from numpy import abs


def plot_time_series(
        link_matrix=None,
        coef_matrix=None,
        var_names=None,
        order=None,
        figsize=None,
        dpi=200,
        label_space_left=0.1,
        label_space_top=0.05,
        label_fontsize=12,
        alpha=0.001
):
    """
    Plot time series graph.

    Parameters
    ----------
    link_matrix : array_like, optional (default: None)
        Matrix of links.
    coef_matrix : array_like
        Matrix of coefficient.
    var_names : list, optional (default: None)
        List of variable names. If None, Xi is used.
    order : list, optional (default: None)
        order of variables from top to bottom.
    figsize : tuple, optional (default: None)
        Size of figure.
    dpi : float, optional (default: 200)
        The resolution of the figure in dots-per-inch.
    label_space_left : float, optional (default: 0.1)
        Fraction of horizontal figure space to allocate left of plot for labels.
    label_space_top : float, optional (default: 0.05)
        Fraction of vertical figure space to allocate top of plot for labels.
    label_fontsize : int, optional (default: 12)
        Fontsize of labels.
    alpha : float, optional (default: 0.001)
        Significance level of link.

    Returns
    -------

    """

    if link_matrix is None and coef_matrix is None:
        raise RuntimeError("link_matrix is None and coef_matrix is None")

    if link_matrix is None:
        link_matrix = np.zeros_like(coef_matrix, dtype=int)
        link_matrix[abs(coef_matrix) >= alpha] = 1

    shape = link_matrix.shape

    assert link_matrix.ndim == 3
    assert shape[0] == shape[1]

    fig = pyplot.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, frame_on=False)

    pyplot.axis("off")

    dim, _, max_lag = link_matrix.shape

    if var_names is None:
        var_names = [fr"$X_{i}$" for i in range(dim)]

    if order is None:
        order = range(dim)

    link_matrix_tsg = np.copy(link_matrix)

    tsg = np.zeros((dim * max_lag, dim * max_lag))

    @lru_cache(128)
    def flat(row, col):
        return row * max_lag + col

    for i, j, tau in np.column_stack(np.where(link_matrix_tsg)):
        for t in range(max_lag):
            if t - tau >= 0:
                tsg[flat(i, t - tau), flat(j, t)] = 1.0

    G = nx.DiGraph(tsg)

    posarray = np.zeros((dim * max_lag, 2))
    for i in range(dim * max_lag):
        posarray[i] = np.array([(i % max_lag), (i // max_lag)])

    pos_tmp = {}

    xmin, ymin = posarray.min(axis=0, initial=0)
    xmax, ymax = posarray.max(axis=0, initial=0)

    for i in range(dim * max_lag):
        pos_tmp[i] = np.array(
            [
                ((i % max_lag) - xmin) / (xmax - xmin),
                ((i // max_lag) - ymin) / (ymax - ymin)
            ]
        )
        pos_tmp[i][np.isnan(pos_tmp[i])] = 0.0

    pos = {}
    for n in range(dim):
        for tau in range(max_lag):
            pos[flat(n, tau)] = pos_tmp[(dim - order[n] - 1) * max_lag + tau]

    for i in range(dim):
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transData)
        ax.text(
            label_space_left,
            pos[order[i] * max_lag][1],
            f"{var_names[order[i]]}",
            fontsize=label_fontsize,
            horizontalalignment="left",
            verticalalignment="center",
            transform=trans,
        )

    for tau in np.arange(max_lag - 1, -1, -1):
        trans = transforms.blended_transform_factory(ax.transData, fig.transFigure)
        if tau == max_lag - 1:
            ax.text(
                pos[tau][0],
                1.0 - label_space_top,
                r"$t$",
                fontsize=label_fontsize,
                horizontalalignment="center",
                verticalalignment="top",
                transform=trans,
            )
        else:
            ax.text(
                pos[tau][0],
                1.0 - label_space_top,
                fr"$t-{max_lag - tau - 1}$",
                fontsize=label_fontsize,
                horizontalalignment="center",
                verticalalignment="top",
                transform=trans,
            )

    nx.draw(G, pos=pos)
    pyplot.show()
