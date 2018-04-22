# -*- coding: utf-8 -*-
from __future__ import print_function
from pprint import pprint

import networkx as nx
from .network_oracle import network_oracle
import numpy as np


def constr(G, e, x):
    u, v = e
    # t is unused here
    if u < v:
        return x[0] - G[u][v]['cost']
    return G[u][v]['cost'] - x[1]


def pconstr(G, e, x):
    u, v = e
    # t is unused here
    if u < v:
        return np.array([1., 0.])
    return np.array([0., -1.])


class optscaling_oracle(network_oracle):

    def __init__(self, G):
        self.G = G
        network_oracle.__init__(self, G, constr, pconstr)

    def __call__(self, x, t):
        g, fj = network_oracle.__call__(self, x)
        if fj > 0.:
            return g, fj, t
        s = x[0] - x[1]
        fj = s - t
        if fj < 0.:
            t = s
            fj = 0.
        return np.array([1., -1.]), fj, t
