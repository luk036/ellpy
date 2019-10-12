# -*- coding: utf-8 -*-
"""
Negative cycle detection for weighed graphs.
1. Support Lazy evalution
"""


def default_get_weight(G, e):
    """[summary]

    Arguments:
        G {[type]} -- [description]
        e {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    u, v = e
    return G[u][v].get('weight', 1)


class negCycleFinder:
    pred = {}

    def __init__(self, G, get_weight=default_get_weight):
        """[summary]

        Arguments:
            G {[type]} -- [description]

        Keyword Arguments:
            get_weight {[type]} -- [description] (default: {default_get_weight})
        """
        self.G = G
        self.get_weight = get_weight
        # self.dist = list(0 for _ in self.G)

    def find_cycle(self):
        """Find a cycle on policy graph

        Arguments:
            G {NetworkX graph}
            pred {dictionary} -- policy graph

        Returns:
            handle -- a start node of the cycle
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

    def relax(self, dist):
        """Perform a updating of dist and pred

        Arguments:
            G {NetworkX graph} -- [description]
            dist {dictionary} -- [description]
            pred {dictionary} -- [description]

        Keyword Arguments:
            weight {str} -- [description]

        Returns:
            [type] -- [description]
        """
        changed = False
        for e in self.G.edges():
            wt = self.get_weight(self.G, e)
            u, v = e
            d = dist[u] + wt
            if dist[v] > d:
                dist[v] = d
                self.pred[v] = u
                changed = True
        return changed

    def find_neg_cycle(self, dist):
        """Perform a updating of dist and pred

        Arguments:
            G {[type]} -- [description]
            dist {dictionary} -- [description]
            pred {dictionary} -- [description]

        Keyword Arguments:
            weight {str} -- [description] (default: {'weight'})

        Returns:
            [type] -- [description]
        """
        # self.dist = list(0 for _ in self.G)
        self.pred = {}
        return self.neg_cycle_relax(dist)

    def neg_cycle_relax(self, dist):
        """[summary]

        Returns:
            [type] -- [description]
        """
        while True:
            changed = self.relax(dist)
            if not changed:
                break
            v = self.find_cycle()
            if v is not None:
                # Will zero cycle be found???
                assert self.is_negative(v, dist)
                return self.cycle_list(v)
        return None

    def cycle_list(self, handle):
        """[summary]

        Arguments:
            handle {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        v = handle
        cycle = list()
        while True:
            u = self.pred[v]
            cycle += [(u, v)]
            v = u
            if v == handle:
                break
        return cycle

    def is_negative(self, handle, dist):
        """[summary]

        Arguments:
            handle {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        v = handle
        # do while loop in C++
        while True:
            u = self.pred[v]
            wt = self.get_weight(self.G, (u, v))
            if dist[v] > dist[u] + wt:
                return True
            v = u
            if v == handle:
                break
        return False
