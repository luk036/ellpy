# -*- coding: utf-8 -*-
from typing import Any, Optional, Tuple

from .neg_cycle import negCycleFinder

Cut = Tuple[Any, float]


class network_oracle:
    """Oracle for Parametric Network Problem:

        find    x, u
        s.t.    u[j] - u[i] ≤ h(e, x)
                ∀ e(i, j) ∈ E

    """
    def __init__(self, G, u, h):
        """[summary]

        Arguments:
            G: a directed graph (V, E)
            u: list or dictionary
            h: function evaluation and gradient
        """
        self.G = G
        self.u = u
        self.h = h
        self.S = negCycleFinder(G)

    def update(self, t: float):
        """[summary]

        Arguments:
            t (float): the best-so-far optimal value
        """
        self.h.update(t)

    def __call__(self, x) -> Optional[Cut]:
        """Make object callable for cutting_plane_feas()

        Arguments:
            x ([type]): [description]

        Returns:
            Optional[Cut]: [description]
        """
        def get_weight(e) -> float:
            """[summary]

            Arguments:
                e ([type]): [description]

            Returns:
                float: [description]
            """
            return self.h.eval(e, x)

        C = self.S.find_neg_cycle(self.u, get_weight)
        if C is not None:
            f = -sum(self.h.eval(e, x) for e in C)
            g = -sum(self.h.grad(e, x) for e in C)
            return g, f
        return None