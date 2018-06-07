# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import time
from ellpy.oracles.profit_oracle import profit_oracle, profit_q_oracle, profit_rb_oracle
from ellpy.cutting_plane import cutting_plane_dc, cutting_plane_q
from ellpy.ell import ell


def benchmark_profit(duration=0.000001):
    """benchmark profit

    Keyword Arguments:
        duration {float} -- [run benchmark duration] (default: {0.000001})
    """

    p, A, k = 20., 40., 30.5
    params = p, A, k
    alpha, beta = 0.1, 0.4
    v1, v2 = 10., 35.
    a = np.array([alpha, beta])
    v = np.array([v1, v2])
    y0 = np.array([0., 0.])  # initial x0
    r = np.array([100., 100.])  # initial ellipsoid (sphere)

    E = ell(r, y0)
    P = profit_oracle(params, a, v)
    _, _, niter, feasible, _ = \
        cutting_plane_dc(P, E, 0.)
    assert feasible
    assert niter == 37

    ui = 1.
    e1 = 0.003
    e2 = 0.007
    e3 = 1.

    E = ell(r, y0)
    P = profit_rb_oracle(params, a, v, (ui, e1, e2, e3))
    _, _, niter, feasible, _ = cutting_plane_dc(P, E, 0.)
    assert feasible
    assert niter == 42

    E = ell(r, y0)
    P = profit_q_oracle(params, a, v)
    _, _, niter, feasible, _ = cutting_plane_q(P, E, 0.)
    assert feasible
    assert niter == 28
    time.sleep(duration)
    return niter


def test_profit():
    result = benchmark_profit()
    assert result == 28