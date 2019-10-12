# -*- coding: utf-8 -*-
import numpy as np

from .network_oracle import network_oracle


def constr(G, e, x):
    """[summary]

    Arguments:
        G {[type]} -- [description]
        e {[type]} -- [description]
        x {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    u, v = e
    cost = G[u][v]['cost']
    assert u != v
    return x[0] - cost if id(u) < id(v) else cost - x[1]


def pconstr(G, e, x):
    """[summary]

    Arguments:
        G {[type]} -- [description]
        e {[type]} -- [description]
        x {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    u, v = e
    assert u != v
    return np.array([1., 0.] if id(u) < id(v) else [0., -1.])


class optscaling_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, G, dist):
        """[summary]

        Arguments:
            G {[type]} -- [description]
        """
        self.G = G
        self.network = network_oracle(G, constr, pconstr, dist)

    def __call__(self, x, t):
        """[summary]

        Arguments:
            x {[type]} -- [description]
            t {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        cut = self.network(x)
        if cut:
            return cut, t
        s = x[0] - x[1]
        fj = s - t
        if fj < 0:
            t = s
            fj = 0.
        return (np.array([1., -1.]), fj), t
