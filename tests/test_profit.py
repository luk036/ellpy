# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import time
from ..oracles.profit_oracle import *
from ..cutting_plane import *
from ..ell import ell


def benchmark_profit(duration=0.000001):
    """benchmark profit
    
    Keyword Arguments:
        duration {float} -- [run benchmark duration] (default: {0.000001})
    """

    p, A, k = 20. , 40. , 30.5
    alpha, beta = 0.1, 0.4
    v1, v2 = 10. , 35.
    a = np.array([alpha, beta])
    v = np.array([v1, v2])
    y0 = np.array([0. , 0.])  # initial x0
    r = np.array([100. , 100.])  # initial ellipsoid (sphere)
    # fmt = '{:f} {} {} {}'

    E = ell(r, y0)
    P = profit_oracle(p, A, a, v, k)
    # yb1, fb, niter, flag, status = cutting_plane_dc(P, E, 0. , 200, 1e-4)
    # print(fmt.format(fb, iter, flag, status))
    _, _, niter, flag, _ = cutting_plane_dc(P, E, 0. , 200, 1e-4)
    assert flag == 1
    assert niter == 37

    ui = 1.
    e1 = 0.003
    e2 = 0.007
    e3 = 1.

    E = ell(r, y0)
    P = profit_rb_oracle(p, A, a, v, k, ui, e1, e2, e3)
    # yb1, fb, niter, flag, status = cutting_plane_dc(P, E, 0. , 200, 1e-4)
    # print(fmt.format(fb, iter, flag, status))
    _, _, niter, flag, _ = cutting_plane_dc(P, E, 0. , 200, 1e-4)
    assert flag == 1
    assert niter == 38

    E = ell(r, y0)
    P = profit_q_oracle(p, A, a, v, k)
    # yb1, fb, niter, flag, status = cutting_plane_q(P, E, 0. , 200, 1e-4)
    # print(fmt.format(fb, iter, flag, status))
    _, _, niter, flag, _ = cutting_plane_q(P, E, 0. , 200, 1e-4)
    assert flag == 1
    assert niter == 28
    time.sleep(duration)
    return niter

def test_profit(benchmark):
    result = benchmark(benchmark_profit)
    assert result == 28
