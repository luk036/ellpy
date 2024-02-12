# -*- coding: utf-8 -*-
from __future__ import print_function

import time
from typing import Optional, Tuple, Union

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
            oracle ([type]): [description]
        """
        self.c = np.array([1.0, -1.0, 1.0])
        F1 = np.array(
            [
                [[-7.0, -11.0], [-11.0, 3.0]],
                [[7.0, -18.0], [-18.0, 8.0]],
                [[-2.0, -8.0], [-8.0, 1.0]],
            ]
        )
        B1 = np.array([[33.0, -9.0], [-9.0, 26.0]])
        F2 = np.array(
            [
                [[-21.0, -11.0, 0.0], [-11.0, 10.0, 8.0], [0.0, 8.0, 5.0]],
                [[0.0, 10.0, 16.0], [10.0, -10.0, -10.0], [16.0, -10.0, 3.0]],
                [[-5.0, 2.0, -17.0], [2.0, -6.0, 8.0], [-17.0, 8.0, 6.0]],
            ]
        )
        B2 = np.array([[14.0, 9.0, 40.0], [9.0, 91.0, 10.0], [40.0, 10.0, 15.0]])
        self.lmi1 = oracle(F1, B1)
        self.lmi2 = oracle(F2, B2)

    def __call__(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """[summary]

        Arguments:
            x (Arr): [description]
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: [description]
        """
        cut = self.lmi1(x)
        if cut:
            return cut, None

        cut = self.lmi2(x)
        if cut:
            return cut, None

        f0 = self.c @ x
        fj = f0 - t
        if fj > 0:
            return (self.c, fj), None
        return (self.c, 0.0), f0


def run_lmi(oracle, duration=0.000001):
    """[summary]

    Arguments:
        oracle ([type]): [description]

    Keyword Arguments:
        duration (float): [description] (default: {0.000001})

    Returns:
        [type]: [description]
    """
    x0 = np.array([0.0, 0.0, 0.0])  # initial x0
    E = ell(10.0, x0)
    P = my_oracle(oracle)
    _, _, ell_info = cutting_plane_dc(P, E, float("inf"))
    time.sleep(duration)

    # fmt = '{:f} {} {} {}'
    # print(fmt.format(fb, niter, feasible, status))
    # print(xb)
    assert ell_info.feasible
    return ell_info.num_iters


def test_lmi_lazy(benchmark):
    """[summary]

    Arguments:
        benchmark ([type]): [description]
    """
    result = benchmark(run_lmi, lmi_oracle)
    assert result == 113


def test_lmi_old(benchmark):
    """[summary]

    Arguments:
        benchmark ([type]): [description]
    """
    result = benchmark(run_lmi, lmi_old_oracle)
    assert result == 113
