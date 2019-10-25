# -*- coding: utf-8 -*-
from typing import Optional, Tuple

from .network_oracle import network_oracle

Cut = Tuple[float, float]


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
        class ratio:
            def __init__(self):
                self.t = None

            def update(self, t: float):
                self.t = t

            def eval(self, G, e, x: float) -> float:
                """[summary]

                Arguments:
                    G {[type]} -- [description]
                    e {[type]} -- [description]
                    x {[type]} -- [description]
                    t {float} -- [description]

                Returns:
                    [type] -- [description]
                """
                u, v = e
                assert u != v
                cost = G[u][v]['cost']
                return x + self.t - cost if u < v else cost - x

            def grad(self, G, e, x: float) -> float:
                """[summary]

                Arguments:
                    G {[type]} -- [description]
                    e {[type]} -- [description]
                    x {[type]} -- [description]
                    t {float} -- [description]

                Returns:
                    [type] -- [description]
                """
                u, v = e
                assert u != v
                return 1. if u < v else -1.

        self.network = network_oracle(G, dist, ratio())

    def update(self, t: float):
        """[summary]

        Arguments:
            t {float} -- [description]
        """
        self.network.update(t)

    def __call__(self, x: float) -> Optional[Cut]:
        """[summary]

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        return self.network(x)
