# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np

from .network3_oracle import network3_oracle

# np.ndarray = np.ndarray
Cut = Tuple[np.ndarray, float]


class optscaling3_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, G, dist):
        """[summary]

        Arguments:
            G {[type]} -- [description]
        """
        # self.G = G
        def constr3(G, e, x: float, t: float) -> float:
            """[summary]

            Arguments:
                G {[type]} -- [description]
                e {[type]} -- [description]
                x {[type]} -- [description]
                t {float} -- [description]

            Returns:
                [type] -- [description]
            """
            u, v = e
            assert u != v
            cost = G[u][v]['cost']
            return x + t - cost if u < v else cost - x

        def pconstr3(G, e, x: float, t: float) -> float:
            """[summary]

            Arguments:
                G {[type]} -- [description]
                e {[type]} -- [description]
                x {[type]} -- [description]
                t {float} -- [description]

            Returns:
                [type] -- [description]
            """
            u, v = e
            assert u != v
            return 1. if u < v else -1.

        self.network3 = network3_oracle(G, dist, constr3, pconstr3)

    def update(self, t: float):
        """[summary]

        Arguments:
            t {float} -- [description]
        """
        self.network3.update(t)

    def __call__(self, x: float):
        """[summary]

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        return self.network3(x)
