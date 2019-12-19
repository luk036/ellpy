# -*- coding: utf-8 -*-
"""
Negative cycle detection for weighed graphs.
1. Support Lazy evalution
"""
from typing import Callable, Optional, Union

D = Union[dict, list]


class negCycleFinder:
    __slots__ = ('G', 'pred', 'cycle')

    def __init__(self, G):
        """[summary]

        Arguments:
            G ([type]): [description]

        Keyword Arguments:
            get_weight (default: {default_get_weight})
        """
        self.G = G
        self.pred: dict = {}
        self.cycle: list = []

        # self.get_weight = get_weight
        # self.dist = list(0 for _ in self.G)

    def find_neg_cycle(self, dist: D, get_weight: Callable) -> Optional[list]:
        """[summary]

        Arguments:
            dist (D): [description]
            get_weight (Callable): [description]

        Returns:
            Optional[list]: [description]
        """
        self.pred = {}
        self.cycle = []
        while self._relax(dist, get_weight):
            v = self._find_cycle()
            if v is not None:
                # Will zero cycle be found???
                assert self._is_negative(v, dist, get_weight)
                self.cycle = self._cycle_list(v)
                break
        return self.cycle

    # private:

    def _find_cycle(self):
        """Find a cycle on policy graph

        Arguments:
            G {NetworkX graph}
            pred (dictionary): policy graph

        Returns:
            handle: a start node of the cycle
        """
        # N = self.G.number_of_nodes()
        # visited = list(N for _ in self.G)
        visited = {}

        for v in self.G:
            if v in visited:
                continue
            u = v
            while True:
                visited[u] = v
                if u not in self.pred:
                    break
                u = self.pred[u]
                # if u is None:
                #    break
                if u in visited:
                    if visited[u] == v:
                        # if self.is_negative(u, dist):
                        return u
                    break
        return None

    def _relax(self, dist, get_weight):
        """Perform a updating of dist and pred

        Arguments:
            G (NetworkX graph): [description]
            dist (dictionary): [description]
            pred (dictionary): [description]

        Keyword Arguments:
            weight (str): [description]

        Returns:
            [type]: [description]
        """
        changed = False
        for e in self.G.edges():
            wt = get_weight(e)
            u, v = e
            d = dist[u] + wt
            if dist[v] > d:
                changed = True
                dist[v] = d
                self.pred[v] = u
        return changed

    def _cycle_list(self, handle):
        """[summary]

        Arguments:
            handle ([type]): [description]

        Returns:
            [type]: [description]
        """
        v = handle
        cycle = []
        while True:
            u = self.pred[v]
            cycle += [(u, v)]
            v = u
            if v == handle:
                break
        return cycle

    def _is_negative(self, handle, dist, get_weight):
        """[summary]

        Arguments:
            handle ([type]): [description]

        Returns:
            [type]: [description]
        """
        v = handle
        # do while loop in C++
        while True:
            u = self.pred[v]
            wt = get_weight((u, v))
            if dist[v] > dist[u] + wt:
                return True
            v = u
            if v == handle:
                break
        return False
