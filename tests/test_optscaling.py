# -*- coding: utf-8 -*-
from __future__ import print_function

import time

import networkx as nx
import numpy as np

from ellpy.cutting_plane import bsearch, bsearch_adaptor, cutting_plane_dc
from ellpy.ell import ell, ell1d
from ellpy.oracles.optscaling3_oracle import optscaling3_oracle
from ellpy.oracles.optscaling_oracle import optscaling_oracle


def vdc(n, base=2):
    """[summary]

    Arguments:
        n {[type]} -- [description]

    Keyword Arguments:
        base {int} -- [description] (default: {2})

    Returns:
        [type] -- [description]
    """
    vdc, denom = 0., 1.
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc


def vdcorput(n, base=2):
    """[summary]

    Arguments:
        n {[type]} -- [description]

    Keyword Arguments:
        base {int} -- [description] (default: {2})

    Returns:
        [type] -- [description]
    """
    return [vdc(i, base) for i in range(n)]


def formGraph(T, pos, eta, seed=None):
    """Form N by N grid of nodes, connect nodes within eta.
        mu and eta are relative to 1/(N-1)

    Arguments:
        T {[type]} -- [description]
        pos {[type]} -- [description]
        eta {[type]} -- [description]

    Keyword Arguments:
        seed {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    if seed:
        np.random.seed(seed)

    N = np.sqrt(T)
    eta = eta / (N - 1)

    # generate perterbed grid positions for the nodes
    pos = dict(enumerate(pos))
    n = len(pos)

    # connect nodes with edges
    G = nx.random_geometric_graph(n, eta, pos=pos)
    G = nx.DiGraph(G)
    # G.add_node('dummy', pos = (0.3, 0.4))
    # G.add_edge('dummy', 1)
    # G.nodemap = {v : i_v for i_v, v in enumerate(G.nodes())}
    return G


N = 75
M = 20
T = N + M
xbase = 2
ybase = 3
x = [i for i in vdcorput(T, xbase)]
y = [i for i in vdcorput(T, ybase)]
pos = zip(x, y)
G = formGraph(T, pos, 1.6, seed=5)
# for u, v in G.edges():
#     h = np.array(G.nodes()[u]['pos']) - np.array(G.nodes()[v]['pos'])
#     G[u][v]['cost'] = np.sqrt(h @ h)

for u, v in G.edges():
    h = np.array(G.nodes()[u]['pos']) - np.array(G.nodes()[v]['pos'])
    G[u][v]['cost'] = np.log(np.sqrt(h @ h))

cmax = max(c for _, _, c in G.edges.data('cost'))
cmin = min(c for _, _, c in G.edges.data('cost'))


def run_optscaling(duration=0.000001):
    """[summary]

    Keyword Arguments:
        duration {float} -- [description] (default: {0.000001})

    Returns:
        [type] -- [description]
    """
    x0 = np.array([cmax, cmin])
    t = cmax - cmin
    E = ell(1.5 * t, x0)
    dist = list(0 for _ in G)
    P = optscaling_oracle(G, dist)
    _, ell_info = cutting_plane_dc(P, E, float('inf'))
    time.sleep(duration)
    # fmt = '{:f} {} {} {}'
    # print(np.exp(xb))
    # print(fmt.format(np.exp(fb), niter, feasible, status))
    assert ell_info.feasible
    return ell_info.num_iters


def run_optscaling3(duration=0.000001):
    """[summary]

    Keyword Arguments:
        duration {float} -- [description] (default: {0.000001})

    Returns:
        [type] -- [description]
    """
    t = cmax - cmin
    Iv = ell1d([cmin, cmax])
    dist = list(0 for _ in G)
    Q = optscaling3_oracle(G, dist)
    P = bsearch_adaptor(Q, Iv)
    _, bs_info = bsearch(P, [0., 1.001 * t])
    time.sleep(duration)
    assert bs_info.feasible
    return bs_info.num_iters


def test_two_variables(benchmark):
    """[summary]

    Arguments:
        benchmark {[type]} -- [description]
    """
    result = benchmark(run_optscaling)
    assert result == 24


def test_binary_search(benchmark):
    """[summary]

    Arguments:
        benchmark {[type]} -- [description]
    """
    result = benchmark(run_optscaling3)
    assert result == 27
