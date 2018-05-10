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

    def __call__(self, x):
        def get_weight(G, e):
            return self.h(G, e, x)

        G = self.G
        S = self.S
        S.get_weight = get_weight
        C = S.find_neg_cycle()

        if C is None:
            return (None, None), 1
        fj = -sum(self.h(G, e, x) for e in C)
        g = -sum(self.ph(G, e, x) for e in C)
        return (g, fj), 0
