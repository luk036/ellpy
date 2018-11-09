# -*- coding: utf-8 -*-
from __future__ import print_function
from pprint import pprint

from networkx.utils import generate_unique_node
import networkx as nx
from ellpy.oracles.neg_cycle import negCycleFinder


def create_test_case1():
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    G[1][2]['weight'] = -5
    newnode = generate_unique_node()
    G.add_edges_from([(newnode, n) for n in G])
    return G


def create_test_case_timing():
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([
        (1, 2, {'weight': 7}),
        (2, 1, {'weight': 0}),
        (2, 3, {'weight': 3}),
        (3, 2, {'weight': 1}),
        (3, 1, {'weight': 2}),
        (1, 3, {'weight': 5})
    ])
    return G


def do_case(G):
    def get_weight(G, e):
        u, v = e
        return G[u][v].get('weight', 1)

    N = negCycleFinder(G, get_weight)
    cycle = N.find_neg_cycle()
    return cycle is not None


def test_cycle():
    G = create_test_case1()
    hasNeg = do_case(G)
    assert hasNeg

    G = nx.path_graph(5, create_using=nx.DiGraph())
    hasNeg = do_case(G)
    assert not hasNeg

    G = create_test_case_timing()
    hasNeg = do_case(G)
    assert not hasNeg
