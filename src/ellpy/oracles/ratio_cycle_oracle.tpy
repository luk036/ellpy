# -*- coding: utf-8 -*-
from typing import Tuple, Union

import numpy as np

from .network_oracle import network_oracle

# np.ndarray = np.ndarray
Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class ratio_cycle_oracle:
    """Oracle for minimum ratio cycle problem.

        This example solves the following convex problem:

            min     t
            s.t.    u[j] - u[i] ≤ mij - sij * x,
                    x ≤ t
    """
    class ratio:
        def __init__(self, G):
            """[summary]

            Arguments:
                G ([type]): [description]
            """
            self.G = G

        def eval(self, e, x: float) -> float:
            """[summary]

            Arguments:
                e ([type]): [description]
                x (float): unknown

            Returns:
                float: function evaluation
            """
            u, v = e
            cost = self.G[u][v]['cost']
            time = self.G[u][v]['time']
            return cost - time * x

        def grad(self, e, x: Arr) -> Arr:
            """[summary]

            Arguments:
                e ([type]): [description]
                x (Arr): (π, ψ) in log scale

            Returns:
                [type]: [description]
            """
            u, v = e
            time = self.G[u][v]['time']
            return -time

    def __init__(self, G, u):
        """Construct a new ratio cycle oracle object

        Arguments:
            G ([type]): [description]
        """
        self.network = network_oracle(G, u, self.ratio(G))

    def __call__(self, x: float, t: float) -> Tuple[Cut, float]:
        """Make object callable for cutting_plane_dc()

        Arguments:
            x (float): unknown
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]

        See also:
            cutting_plane_dc
        """
        fj = x - t
        if fj >= 0.:
            return (1. fj), t

        cut = self.network(x)
        if cut:
            return cut, t

        return (1., 0), x
