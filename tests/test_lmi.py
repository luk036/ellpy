# -*- coding: utf-8 -*-
from __future__ import print_function

import time
from typing import Tuple, Union

import numpy as np

from ellpy.cutting_plane import cutting_plane_dc
from ellpy.ell import ell
from ellpy.oracles.lmi_old_oracle import lmi_old_oracle
from ellpy.oracles.lmi_oracle import lmi_oracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class my_oracle:
    def __init__(self, oracle):
        """[summary]

        Arguments:
            oracle {[type]} -- [description]
        """
        self.c = np.array([1., -1., 1.])
        F1 = np.array([[[-7., -11.], [-11., 3.]], [[7., -18.], [-18., 8.]],
                       [[-2., -8.], [-8., 1.]]])
        B1 = np.array([[33., -9.], [-9., 26.]])
        F2 = np.array([[[-21., -11., 0.], [-11., 10., 8.], [0., 8., 5.]],
                       [[0., 10., 16.], [10., -10., -10.], [16., -10., 3.]],
                       [[-5., 2., -17.], [2., -6., 8.], [-17., 8., 6.]]])
        B2 = np.array([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]])
        self.lmi1 = oracle(F1, B1)
        self.lmi2 = oracle(F2, B2)

    def __call__(self, x: Arr, t: float) -> Tuple[Cut, float]:
        """[summary]

        Arguments:
            x {Arr} -- [description]
            t {float} -- the best-so-far optimal value

        Returns:
            Tuple[Cut, float] -- [description]
        """
        f0 = self.c @ x
        fj = f0 - t
        if fj > 0:
            return (self.c, fj), t

        cut = self.lmi1(x)
        if cut:
            return cut, t

        cut = self.lmi2(x)
        if cut:
            return cut, t
        return (self.c, 0.), f0


def run_lmi(oracle, duration=0.000001):
    """[summary]

    Arguments:
        oracle {[type]} -- [description]

    Keyword Arguments:
        duration {float} -- [description] (default: {0.000001})

    Returns:
        [type] -- [description]
    """
    x0 = np.array([0., 0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_oracle(oracle)
    _, ell_info = cutting_plane_dc(P, E, float('inf'))
    time.sleep(duration)

    # fmt = '{:f} {} {} {}'
    # print(fmt.format(fb, niter, feasible, status))
    # print(xb)
    assert ell_info.feasible
    return ell_info.num_iters


def test_lmi_lazy(benchmark):
    """[summary]

    Arguments:
        benchmark {[type]} -- [description]
    """
    result = benchmark(run_lmi, lmi_oracle)
    assert result == 115


def test_lmi_old(benchmark):
    """[summary]

    Arguments:
        benchmark {[type]} -- [description]
    """
    result = benchmark(run_lmi, lmi_old_oracle)
    assert result == 115
