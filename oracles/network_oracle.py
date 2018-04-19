# -*- coding: utf-8 -*-
import networkx as nx
from .neg_cycle import negCycleFinder
import numpy as np


class network_oracle:
    """
        Oracle for Linear Matrix Inequality constraint
            F * x <= B
        Or
            (B - F * x) must be a semidefinte matrix
    """
    def __init__(self, G, h, ph):
        self.G = G
        self.h = h
        self.ph = ph  # partial derivative of h w.r.t x
        self.S = negCycleFinder(G)

    def __call__(self, x, t):
        def get_weight(G, e):
            return self.h(G, e, x, t)

        G = self.G
        S = self.S
        S.get_weight = get_weight

        n = len(x)

        for (u, v) in G.edges:
            G[u][v]['weight'] = self.h(G, (u, v), x, t)

        C = S.find_neg_cycle()
        if C is None:
            return np.zeros(n), -1

        # fj = -sum(G[u][v]['weight'] for u, v in C)
        # g = -sum(self.ph(G, (u, v), x, t) for u, v in C)
        fj = 0.
        g = np.zeros(n)
        for u, v in C:
            fj -= G[u][v]['weight']
            g -= self.ph(G, (u, v), x, t)

        return g, fj
