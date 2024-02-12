# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline

from .corr_oracle import construct_distance_matrix

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


def mono_oracle(x):
    """[summary]

    Arguments:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    # monotonic decreasing constraint
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        fj = x[i + 1] - x[i]
        if fj > 0:
            g[i] = -1.0
            g[i + 1] = 1.0
            return g, fj


class mono_decreasing_oracle2:
    """[summary]

    Returns:
        [type]: [description]
    """

    def __init__(self, basis):
        """[summary]

        Arguments:
            basis ([type]): [description]
        """
        self.basis = basis

    def __call__(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """[summary]

        Arguments:
            x (Arr): [description]
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: [description]
        """
        # monotonic decreasing constraint
        n = len(x)
        g = np.zeros(n)
        cut = mono_oracle(x[:-1])
        if cut:
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.0
            return (g, fj), None
        return self.basis(x, t)


def corr_bspline(Y, s, m, oracle, corr_core):
    """[summary]

    Arguments:
        Y ([type]): [description]
        s ([type]): [description]
        m ([type]): [description]
        oracle ([type]): [description]
        corr_core ([type]): [description]

    Returns:
        [type]: [description]
    """
    Sig, t, k = generate_bspline_info(s, m)
    Pb = oracle(Sig, Y)
    P = mono_decreasing_oracle2(Pb)
    c, num_iters, feasible = corr_core(Y, m, P)
    return BSpline(t, c, k), num_iters, feasible


def generate_bspline_info(s, m):
    """[summary]

    Arguments:
        s ([type]): [description]
        m ([type]): [description]

    Returns:
        [type]: [description]
    """
    k = 2  # quadratic bspline
    h = s[-1] - s[0]
    d = np.sqrt(h @ h)
    t = np.linspace(0, d * 1.2, m + k + 1)
    spls = []
    for i in range(m):
        coeff = np.zeros(m)
        coeff[i] = 1
        spls += [BSpline(t, coeff, k)]
    D = construct_distance_matrix(s)
    Sig = []
    for i in range(m):
        Sig += [spls[i](D)]
    return Sig, t, k
