# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import numpy as np
from ..oracles.lmi_oracle import lmi_oracle
from ..oracles.lmi_old_oracle import lmi_old_oracle
from ..cutting_plane import cutting_plane_dc
from ..ell import ell


class my_oracle:
    def __init__(self, oracle):
        self.c = np.array([1., -1., 1.])
        F1 = np.array([[[-7., -11.], [-11., 3.]],
                       [[7., -18.], [-18., 8.]],
                       [[-2., -8.], [-8., 1.]]])
        B1 = np.array([[33., -9.], [-9., 26.]])
        F2 = np.array([[[-21., -11., 0.], [-11., 10., 8.], [0., 8., 5.]],
                       [[0., 10., 16.], [10., -10., -10.], [16., -10., 3.]],
                       [[-5., 2., -17.], [2., -6., 8.], [-17., 8., 6.]]])
        B2 = np.array([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]])
        self.lmi1 = oracle(F1, B1)
        self.lmi2 = oracle(F2, B2)

    def __call__(self, x, t):
        f0 = np.dot(self.c, x)
        fj = f0 - t
        if fj > 0.:
            return (self.c, fj), t

        cut, feasible = self.lmi1(x)
        if not feasible:
            return cut, t

        cut, feasible = self.lmi2(x)
        if not feasible:
            return cut, t
        return (self.c, 0.), f0


def run_lmi(oracle, duration=0.000001):
    x0 = np.array([0., 0., 0.])  # initial x0
    fmt = '{:f} {} {} {}'

    E = ell(10., x0)
    P = my_oracle(oracle)
    xb, fb, niter, feasible, status = cutting_plane_dc(P, E, 100.)
    print(fmt.format(fb, niter, feasible, status))
    print(xb)
    assert feasible
    time.sleep(duration)
    return niter


def test_lmi_lazy(benchmark):
    result = benchmark(run_lmi, lmi_oracle)
    assert result == 115


def test_lmi_old(benchmark):
    result = benchmark(run_lmi, lmi_old_oracle)
    assert result == 115
