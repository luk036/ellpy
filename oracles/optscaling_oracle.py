# -*- coding: utf-8 -*-
import networkx as nx
from .network_oracle import network_oracle
import numpy as np


def constr(G, e, x):
    u, v = e
    if u < v:
        return x[0] - G[u][v]['cost']
    else:
        return G[u][v]['cost'] - x[1]


def pconstr(G, e, x):
    u, v = e
    if u < v:
        return np.array([1., 0.])
    else:
        return np.array([0., -1.])


class optscaling_oracle:

    def __init__(self, G):
        self.network = network_oracle(G, constr, pconstr)

    def __call__(self, x, t):
        cut, feasible = self.network(x)
        if not feasible:
            return cut, t
        s = x[0] - x[1]
        fj = s - t
        if fj < 0.:
            t = s
            fj = 0.
        return (np.array([1., -1.]), fj), t
