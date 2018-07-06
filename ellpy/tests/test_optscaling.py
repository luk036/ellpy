# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import numpy as np
import networkx as nx
from ellpy.cutting_plane import cutting_plane_dc, bsearch, bsearch_adaptor
from ellpy.oracles.optscaling_oracle import optscaling_oracle
from ellpy.oracles.optscaling3_oracle import optscaling3_oracle
from ellpy.ell import ell, ell1d


def vdc(n, base=2):
    vdc, denom = 0., 1.
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc


def vdcorput(n, base=2):
    '''
    n - number of vectors
    base - seeds
    '''
    return [vdc(i, base) for i in range(n)]


def formGraph(T, pos, eta, seed=None):
    ''' Form N by N grid of nodes, connect nodes within eta.
        mu and eta are relative to 1/(N-1)
    '''
    if seed is not None:
        np.random.seed(seed)

    N = np.sqrt(T)
    eta = eta/(N-1)

    # generate perterbed grid positions for the nodes
    pos = dict(enumerate(pos))
    n = len(pos)

    # connect nodes with edges
    G = nx.random_geometric_graph(n, eta, pos=pos)
    G = nx.DiGraph(G)
    return G


N = 75
M = 20
T = N+M
xbase = 2
ybase = 3
x = [i for i in vdcorput(T, xbase)]
y = [i for i in vdcorput(T, ybase)]
pos = zip(x, y)
G = formGraph(T, pos, 1.6, seed=5)
# Add a sink, connect all spareTSV to it.
for u, v in G.edges():
    h = np.array(G.node[u]['pos']) - np.array(G.node[v]['pos'])
    G[u][v]['cost'] = np.sqrt(np.dot(h, h))

for u, v in G.edges():
    G[u][v]['cost'] = np.log(abs(G[u][v]['cost']))

cmax = max(c for _, _, c in G.edges.data('cost'))
cmin = min(c for _, _, c in G.edges.data('cost'))


def run_optscaling(duration=0.000001):
    x0 = np.array([cmax, cmin])
    t = cmax - cmin
    E = ell(1.5*t, x0)
    P = optscaling_oracle(G)
    xb, fb, niter, feasible, status = cutting_plane_dc(P, E, 1.001*t)
    time.sleep(duration)
    fmt = '{:f} {} {} {}'
    print(np.exp(xb))
    print(fmt.format(np.exp(fb), niter, feasible, status))
    assert feasible
    return niter


def run_optscaling3(duration=0.000001):
    t = cmax - cmin
    I = ell1d([cmin, cmax])
    Q = optscaling3_oracle(G)
    P = bsearch_adaptor(Q, I)
    _, niter, feasible = bsearch(P, [0., 1.001*t])
    time.sleep(duration)
    assert feasible
    return niter


def test_two_variables(benchmark):
    result = benchmark(run_optscaling)
    assert result == 26


def test_binary_search(benchmark):
    result = benchmark(run_optscaling3)
    assert result == 27
