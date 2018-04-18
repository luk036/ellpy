# -*- coding: utf-8 -*-
"""
Negative cycle detection for weighed graphs.
"""
from __future__ import print_function
from pprint import pprint

from collections import deque
import networkx as nx
from networkx.utils import generate_unique_node


def set_default(G, weight, value):
    for (u, v) in G.edges:
        if G[u][v].get(weight, None) == None:
            G[u][v][weight] = value


def create_test_case1():
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    G[1][2]['weight'] = -5
    newnode = generate_unique_node()
    G.add_edges_from([(newnode, n) for n in G])
    return G


class negCycleFinder:

    def __init__(self, G):
        """Relaxation loop for Bellman–Ford algorithm

        Parameters
        ----------
        G : NetworkX graph
        """

        self.G = G
        self.dist = {v: 0 for v in G}
        self.pred = {v: None for v in G}
        #self.dist = dist.copy()
        #self.pred = pred.copy()

    def find_cycle(self, weight):
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
                if u == None:
                    break
                if visited[u] != None:
                    if visited[u] == v:
                        if self.is_negative(u, weight):
                            return u
                    break
        return None

    def relax(self, weight):
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
        for (u, v, wt) in self.G.edges.data(weight):
            d = self.dist[u] + wt
            if self.dist[v] > d:
                self.dist[v] = d
                self.pred[v] = u
                changed = True
        return changed

    def find_neg_cycle(self, weight='weight'):
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
        G = self.G
        self.dist = {v: 0. for v in G}
        self.pred = {v: None for v in G}
        set_default(G, weight, 1)
        c = self.neg_cycle_relax(weight)
        return c

    def neg_cycle_relax(self, weight='weight'):
        G = self.G
        #self.dist = {v: 0. for v in G}
        self.pred = {v: None for v in G}

        while True:
            changed = self.relax(weight)
            if changed:
                v = self.find_cycle(weight)
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
            cycle += {(u, v)}
            v = u
            if v == handle:
                break
        return cycle

    def is_negative(self, handle, weight):
        v = handle
        while True:
            u = self.pred[v]
            if self.dist[v] > self.dist[u] + self.G[u][v][weight]:
                return True
            v = u
            if v == handle:
                break
        return False
