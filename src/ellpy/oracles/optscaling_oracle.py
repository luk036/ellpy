# -*- coding: utf-8 -*-
from typing import Tuple, Union

import numpy as np

from .network_oracle import network_oracle

# np.ndarray = np.ndarray
Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class optscaling_oracle:
    """Oracle for Optimal Matrix Scaling

    This example is taken from[Orlin and Rothblum, 1985]

        min     π/ψ
        s.t.    ψ ≤ u[i] * |aij| * u[j]^{-1} ≤ π,
                ∀ aij != 0,
                π, ψ, u, positive
    """
    class ratio:
        def eval(self, G, e, x: Arr) -> float:
            """[summary]

            Arguments:
                G {[type]} -- [description]
                e {[type]} -- [description]
                x {Arr} -- (π, ψ) in log scale

            Returns:
                float -- function evaluation
            """
            u, v = e
            cost = G[u][v]['cost']
            assert u != v
            return x[0] - cost if u < v else cost - x[1]

        def grad(self, G, e, x: Arr) -> Arr:
            """[summary]

            Arguments:
                G {[type]} -- [description]
                e {[type]} -- [description]
                x {Arr} -- (π, ψ) in log scale

            Returns:
                [type] -- [description]
            """
            u, v = e
            assert u != v
            return np.array([1., 0.] if u < v else [0., -1.])

    def __init__(self, G, u):
        """Construct a new optscaling oracle object

        Arguments:
            G {[type]} -- [description]
        """
        self.network = network_oracle(G, u, self.ratio())

    def __call__(self, x: Arr, t: float) -> Tuple[Cut, float]:
        """Make object callable for cutting_plane_dc()

        Arguments:
            x {Arr} -- (π, ψ) in log scale
            t {float} -- the best-so-far optimal value

        Returns:
            Tuple[Cut, float]

        See also:
            cutting_plane_dc
        """
        cut = self.network(x)
        if cut:
            return cut, t

        s = x[0] - x[1]
        fj = s - t
        if fj < 0.:
            t = s
            fj = 0.
        return (np.array([1., -1.]), fj), t
