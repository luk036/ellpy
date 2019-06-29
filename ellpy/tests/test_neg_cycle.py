# -*- coding: utf-8 -*-
from __future__ import print_function

from networkx.utils import generate_unique_node
import networkx as nx
from ellpy.oracles.neg_cycle import negCycleFinder


class SimpleDiGraph(nx.DiGraph):
    nodemap = {}


def create_test_case1():
    """[summary]

    Returns:
        [type] -- [description]
    """
    G = SimpleDiGraph(nx.cycle_graph(5, create_using=nx.DiGraph()))
    G[1][2]['weight'] = -5
    # newnode = generate_unique_node()
    G.add_edges_from([(5, n) for n in G])
    G.nodemap = range(6)
    return G


def create_test_case_timing():
    """[summary]

    Returns:
        [type] -- [description]
    """
    G = SimpleDiGraph()
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
    G.nodemap = {v: i_v for i_v, v in enumerate(nodelist)}
    return G


def do_case(G):
    """[summary]

    Arguments:
        G {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    def get_weight(G, e):
        """[summary]

        Arguments:
            G {[type]} -- [description]
            e {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        u, v = e
        return G[u][v].get('weight', 1)

    N = negCycleFinder(G, get_weight)
    cycle = N.find_neg_cycle()
    return cycle is not None


def test_cycle():
    """[summary]
    """
    G = create_test_case1()
    hasNeg = do_case(G)
    assert hasNeg

    G = SimpleDiGraph(nx.path_graph(5, create_using=nx.DiGraph()))
    G.nodemap = range(G.number_of_nodes())
    hasNeg = do_case(G)
    assert not hasNeg

    G = create_test_case_timing()
    hasNeg = do_case(G)
    assert not hasNeg
