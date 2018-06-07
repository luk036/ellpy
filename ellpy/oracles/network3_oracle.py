# -*- coding: utf-8 -*-
from .neg_cycle import negCycleFinder


class network3_oracle:

    def __init__(self, G, f, p):
        self.G = G
        self.f = f
        self.p = p  # partial derivative of f w.r.t x
        self.S = negCycleFinder(G)
        self.t = None

    def update(self, t):
        self.t = t

    def __call__(self, x):
        def get_weight(G, e):
            return self.f(G, e, x, self.t)

        self.S.get_weight = get_weight
        C = self.S.find_neg_cycle()
        if C is not None:
            f = -sum(self.f(self.G, e, x, self.t) for e in C)
            g = -sum(self.p(self.G, e, x, self.t) for e in C)
            return (g, f), False

        return None, True
