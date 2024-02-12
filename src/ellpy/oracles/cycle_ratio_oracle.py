# -*- coding: utf-8 -*-
from typing import Any, Tuple, Union

import numpy as np

from .network_oracle import network_oracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, Any]


class cycle_ratio_oracle:
    """Oracle for minimum ratio cycle problem.

    This example solves the following convex problem:

        max     t
        s.t.    u[j] - u[i] ≤ mij - sij * x,
                t ≤ x

    where sij is not necessarily positive.
    The problem could be unbounded???

    """

    class ratio:
        def __init__(self, G):
            """[summary]

            Arguments:
                G: [description]
            """
            self.G = G

        def eval(self, e, x):
            """[summary]

            Arguments:
                e ([type]): [description]
                x: unknown

            Returns:
                Any: function evaluation
            """
            u, v = e
            cost = self.G[u][v]["cost"]
            time = self.G[u][v]["time"]
            return cost - time * x

        def grad(self, e, x):
            """[summary]

            Arguments:
                e ([type]): [description]
                x (Arr): (π, ψ) in log scale

            Returns:
                [type]: [description]
            """
            u, v = e
            time = self.G[u][v]["time"]
            return -time

    def __init__(self, G, u):
        """Construct a new ratio cycle oracle object

        Arguments:
            G ([type]): [description]
        """
        self.network = network_oracle(G, u, self.ratio(G))

    def __call__(self, x, t) -> Tuple[Cut, Any]:
        """Make object callable for cutting_plane_dc()

        Arguments:
            x (Any): unknown
            t (Any): the best-so-far optimal value

        Returns:
            Tuple[Cut, Any]

        See also:
            cutting_plane_dc
        """
        fj = t - x
        if fj >= 0:
            return (-1, fj), None

        cut = self.network(x)
        if cut:
            return cut, None

        return (-1, 0.0), x
