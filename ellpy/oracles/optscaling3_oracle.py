# -*- coding: utf-8 -*-
from .network3_oracle import network3_oracle


def constr3(G, e, x, t):
    u, v = e
    if u <= v:
        return x + t - G[u][v]['cost']
    return G[u][v]['cost'] - x


def pconstr3(G, e, x, t):
    u, v = e
    if u <= v:
        return 1.
    return -1.


class optscaling3_oracle:

    def __init__(self, G):
        self.G = G
        self.network3 = network3_oracle(G, constr3, pconstr3)

    def update(self, t):
        self.network3.update(t)

    def __call__(self, x):
        return self.network3(x)
