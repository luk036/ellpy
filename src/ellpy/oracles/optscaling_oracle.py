# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np

from .network_oracle import network_oracle

# np.ndarray = np.ndarray
Cut = Tuple[np.ndarray, float]


class optscaling_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, G, dist):
        """[summary]

        Arguments:
            G {[type]} -- [description]
        """

        def constr(G, e, x):
            """[summary]

            Arguments:
                G {[type]} -- [description]
                e {[type]} -- [description]
                x {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            u, v = e
            cost = G[u][v]['cost']
            assert u != v
            return x[0] - cost if u < v else cost - x[1]

        def pconstr(G, e, x):
            """[summary]

            Arguments:
                G {[type]} -- [description]
                e {[type]} -- [description]
                x {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            u, v = e
            assert u != v
            return np.array([1., 0.] if u < v else [0., -1.])

        self.network = network_oracle(G, dist, constr, pconstr)

    def __call__(self, x: np.ndarray, t: float) -> Tuple[Cut, float]:
        """[summary]

        Arguments:
            x {np.ndarray} -- [description]
            t {float} -- [description]

        Returns:
            Tuple[Cut, float] -- [description]
        """
        cut = self.network(x)
        if cut:
            return cut, t
        s = x[0] - x[1]
        fj = s - t
        if fj < 0:
            t = s
            fj = 0.
        return (np.array([1., -1.]), fj), t
