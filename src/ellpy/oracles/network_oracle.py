# -*- coding: utf-8 -*-
from typing import Any, Optional, Tuple

from netoptim.neg_cycle import negCycleFinder

Cut = Tuple[Any, float]


class network_oracle:
    """Oracle for Parametric Network Problem:

        find    x, u
        s.t.    u[j] − u[i] ≤ h(e, x)
                ∀ e(i, j) ∈ E

    """
    def __init__(self, G, u, h):
        """[summary]

        Arguments:
            G: a directed graph (V, E)
            u: list or dictionary
            h: function evaluation and gradient
        """
        self._G = G
        self._u = u
        self._h = h
        self._S = negCycleFinder(G)

    def update(self, t):
        """[summary]

        Arguments:
            t (float): the best-so-far optimal value
        """
        self._h.update(t)

    def __call__(self, x) -> Optional[Cut]:
        """Make object callable for cutting_plane_feas()

        Arguments:
            x ([type]): [description]

        Returns:
            Optional[Cut]: [description]
        """
        def get_weight(e):
            """[summary]

            Arguments:
                e ([type]): [description]

            Returns:
                Any: [description]
            """
            return self._h.eval(e, x)

        for Ci in self._S.find_neg_cycle(self._u, get_weight):
            f = -sum(self._h.eval(e, x) for e in Ci)
            g = -sum(self._h.grad(e, x) for e in Ci)
            return g, f
        return None
