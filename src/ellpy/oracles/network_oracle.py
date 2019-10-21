# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np

from .neg_cycle import negCycleFinder

# np.ndarray = np.ndarray
Cut = Tuple[np.ndarray, float]


class network_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, G, dist, f, p):
        """initization

        Arguments:
            G {Graph's node} -- [description]
            f {function} -- [description]
            p {gradient} -- [description]
        """
        self.G = G
        self.dist = dist
        self.f = f
        self.p = p  # partial derivative of f w.r.t x
        self.S = negCycleFinder(G)

    def __call__(self, x: np.ndarray) -> Optional[Cut]:
        """[summary]

        Arguments:
            x {np.ndarray} -- [description]

        Returns:
            Optional[Cut] -- [description]
        """
        def get_weight(G, e):
            """get weight

            Arguments:
                self {[type]} -- [description]
                e {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            return self.f(G, e, x)

        # self.S.get_weight = get_weight
        C = self.S.find_neg_cycle(self.dist, get_weight)
        if C is None:
            return None
        f = -sum(self.f(self.G, e, x) for e in C)
        g = -sum(self.p(self.G, e, x) for e in C)
        return g, f
