# -*- coding: utf-8 -*-
import networkx as nx
from .network3_oracle import network3_oracle
import numpy as np


def constr3(G, e, x, t):
    u, v = e
    if u <= v:
        return x + t - G[u][v]['cost']
    else:
        return G[u][v]['cost'] - x


def pconstr3(G, e, x, t):
    u, v = e
    if u <= v:
        return 1.
    else:
        return -1.


class optscaling3_oracle:

    def __init__(self, G):
        self.G = G
        self.network3 = network3_oracle(G, constr3, pconstr3)

    def update(self, t):
        self.network3.update(t)

    def __call__(self, x):
        for (u, v) in self.G.edges():
            if u != v:
                continue
            fj = x - self.G[u][v]['cost']
            if fj > 0.:
                return (1., fj), False

        return self.network3(x)
