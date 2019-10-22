# -*- coding: utf-8 -*-
from typing import Any, Optional, Tuple

from .neg_cycle import negCycleFinder

Cut = Tuple[Any, float]


class network3_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    t = None

    def __init__(self, G, dist, f, p):
        """[summary]

        Arguments:
            G {[type]} -- [description]
            f {[type]} -- [description]
            p {[type]} -- [description]
        """
        self.G = G
        self.dist = dist
        self.f = f
        self.p = p  # partial derivative of f w.r.t x
        self.S = negCycleFinder(G)

    def update(self, t: float):
        """[summary]

        Arguments:
            t {float} -- [description]
        """
        self.t = t

    def __call__(self, x) -> Optional[Cut]:
        """[summary]

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        def get_weight(G, e):
            """[summary]

            Arguments:
                G {[type]} -- [description]
                e {[type]} -- [description]

            Returns:
                [type] -- [description]
            """
            return self.f(G, e, x, self.t)

        # self.S.get_weight = get_weight
        C = self.S.find_neg_cycle(self.dist, get_weight)
        if C is None:
            return None

        f = -sum(self.f(self.G, e, x, self.t) for e in C)
        g = -sum(self.p(self.G, e, x, self.t) for e in C)
        return g, f
