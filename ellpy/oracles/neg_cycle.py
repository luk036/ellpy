# -*- coding: utf-8 -*-
"""
Negative cycle detection for weighed graphs.
"""

def default_get_weight(G, e):
    u, v = e
    return G[u][v].get('weight', 1)


class negCycleFinder:

    def __init__(self, G, get_weight=default_get_weight):
        self.G = G
        self.dist = {v: 0 for v in G}
        self.pred = {v: None for v in G}
        self.get_weight = get_weight

    def find_cycle(self):
        """Find a cycle on policy graph

        Arguments:
            G {NetworkX graph}
            pred {dictionary} -- policy graph

        Returns:
            handle -- a start node of the cycle
        """

        visited = {v: None for v in self.G}
        for v in self.G:
            if visited[v] != None:
                continue
            u = v
            while True:
                visited[u] = v
                u = self.pred[u]
                if u is None:
                    break
                if visited[u] != None:
                    if visited[u] == v:
                        if self.is_negative(u):
                            # should be "yield u"
                            return u
                    break
        return None

    def relax(self):
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
        for (u, v) in self.G.edges:
            wt = self.get_weight(self.G, (u, v))
            d = self.dist[u] + wt
            if self.dist[v] > d:
                self.dist[v] = d
                self.pred[v] = u
                changed = True
        return changed

    def find_neg_cycle(self):
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
        self.dist = {v: 0. for v in self.G}
        self.pred = {v: None for v in self.G}
        return self.neg_cycle_relax()

    def neg_cycle_relax(self):
        self.pred = {v: None for v in self.G}

        while True:
            changed = self.relax()
            if changed:
                v = self.find_cycle()
                if v != None:
                    return self.cycle_list(v)
            else:
                break
        return None

    def cycle_list(self, handle):
        v = handle
        cycle = list()
        while True:
            u = self.pred[v]
            cycle += [(u, v)]
            v = u
            if v == handle:
                break
        return cycle

    def is_negative(self, handle):
        v = handle
        while True:
            u = self.pred[v]
            wt = self.get_weight(self.G, (u, v))
            if self.dist[v] > self.dist[u] + wt:
                return True
            v = u
            if v == handle:
                break
        return False
