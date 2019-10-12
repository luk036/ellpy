# -*- coding: utf-8 -*-
from .network3_oracle import network3_oracle


def constr3(G, e, x, t):
    """[summary]

    Arguments:
        G {[type]} -- [description]
        e {[type]} -- [description]
        x {[type]} -- [description]
        t {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    u, v = e
    assert u != v
    cost = G[u][v]['cost']
    return x + t - cost if id(u) < id(v) else cost - x


def pconstr3(G, e, x, t):
    """[summary]

    Arguments:
        G {[type]} -- [description]
        e {[type]} -- [description]
        x {[type]} -- [description]
        t {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    u, v = e
    assert u != v
    return 1. if id(u) < id(v) else -1.


class optscaling3_oracle:
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
        self.network3 = network3_oracle(G, constr3, pconstr3, dist)

    def update(self, t):
        """[summary]

        Arguments:
            t {[type]} -- [description]
        """
        self.network3.update(t)

    def __call__(self, x):
        """[summary]

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        return self.network3(x)
