# -*- coding: utf-8 -*-
from __future__ import print_function

from fractions import Fraction

import networkx as nx

from ellpy.cutting_plane import cutting_plane_dc
from ellpy.ell import ell1d
from ellpy.oracles.cycle_ratio_oracle import cycle_ratio_oracle


def set_default(G: nx.Graph, weight, value):
    """[summary]

    Arguments:
        G (nx.Graph): directed graph
        weight ([type]): [description]
        value ([type]): [description]
    """
    for u, v in G.edges:
        if G[u][v].get(weight, None) is None:
            G[u][v][weight] = value


def create_test_case1():
    """[summary]

    Returns:
        [type]: [description]
    """
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    G[1][2]['weight'] = -5
    G.add_edges_from([(5, n) for n in G])
    return G


def create_test_case_timing():
    """[summary]

    Returns:
        [type]: [description]
    """
    G = nx.DiGraph()
    nodelist = ['a1', 'a2', 'a3']
    G.add_nodes_from(nodelist)
    G.add_edges_from([
        ('a1', 'a2', {'weight': 7}),
        ('a2', 'a1', {'weight': 0}),
        ('a2', 'a3', {'weight': 3}),
        ('a3', 'a2', {'weight': 1}),
        ('a3', 'a1', {'weight': 2}),
        ('a1', 'a3', {'weight': 5})
    ])
    return G


def test_cycle_ratio():
    G = create_test_case1()
    set_default(G, 'time', 1)
    set_default(G, 'cost', 1)
    G[1][2]['cost'] = 5
    dist = list(Fraction(0, 1) for _ in G)

    E = ell1d([Fraction(-100, 1), Fraction(100, 1)])
    assert E._xc == Fraction(0, 1)
    assert E._r == Fraction(100, 1)
    P = cycle_ratio_oracle(G, dist)
    _, r, ell_info = cutting_plane_dc(P, E, Fraction(-1000000, 1))
    assert ell_info.feasible
    # assert r == Fraction(9, 5)


def test_cycle_ratio_timing():
    G = create_test_case_timing()
    set_default(G, 'time',  1)
    G['a1']['a2']['cost'] = 7
    G['a2']['a1']['cost'] = -1
    G['a2']['a3']['cost'] = 3
    G['a3']['a2']['cost'] = 0
    G['a3']['a1']['cost'] = 2
    G['a1']['a3']['cost'] = 4
    # make sure no parallel edges in above!!!
    dist = {v: Fraction(0, 1) for v in G}

    E = ell1d([Fraction(-100, 1), Fraction(100, 1)])
    assert E._xc == Fraction(0, 1)
    assert E._r == Fraction(100, 1)
    P = cycle_ratio_oracle(G, dist)
    _, r, ell_info = cutting_plane_dc(P, E, Fraction(-1000000, 1))
    assert ell_info.feasible
    # assert r == Fraction(1, 1)
