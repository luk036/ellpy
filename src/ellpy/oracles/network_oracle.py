# -*- coding: utf-8 -*-
from typing import Any, Optional, Tuple, Callable
from .neg_cycle import negCycleFinder

Cut = Tuple[Any, float]


class network_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, G, dist, h):
        """[summary]

        Arguments:
            G {[type]} -- [description]
            f {[type]} -- [description]
            p {[type]} -- [description]
        """
        self.G = G
        self.dist = dist
        self.h = h
        self.S = negCycleFinder(G)

    def update(self, t):
        """[summary]

        Arguments:
            t {float} -- [description]
        """
        self.h.update(t)

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
            return self.h.eval(G, e, x)

        C = self.S.find_neg_cycle(self.dist, get_weight)
        if C is None:
            return None

        f = -sum(self.h.eval(self.G, e, x) for e in C)
        g = -sum(self.h.grad(self.G, e, x) for e in C)
        return g, f
