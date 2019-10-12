# -*- coding: utf-8 -*-
from .neg_cycle import negCycleFinder


class network3_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    t = None

    def __init__(self, G, f, p, dist):
        """[summary]

        Arguments:
            G {[type]} -- [description]
            f {[type]} -- [description]
            p {[type]} -- [description]
        """
        self.G = G
        self.f = f
        self.p = p  # partial derivative of f w.r.t x
        self.S = negCycleFinder(G)
        self.dist = dist

    def update(self, t):
        """[summary]

        Arguments:
            t {[type]} -- [description]
        """
        self.t = t

    def __call__(self, x):
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

        self.S.get_weight = get_weight
        C = self.S.find_neg_cycle(self.dist)
        if C is None:
            return None

        f = -sum(self.f(self.G, e, x, self.t) for e in C)
        g = -sum(self.p(self.G, e, x, self.t) for e in C)
        return g, f
