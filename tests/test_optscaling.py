# -*- coding: utf-8 -*-
from __future__ import print_function

# import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from ..cutting_plane import cutting_plane_dc
from ..oracles.optscaling_oracle import optscaling_oracle
from ..ell import ell


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


# if __name__ == "__main__":
def test_optscaling():
    N = 15
    M = 4
#    r = 4

    T = N+M
    xbase = 2
    ybase = 3
    x = [i for i in vdcorput(T, xbase)]
    y = [i for i in vdcorput(T, ybase)]
    pos = zip(x, y)
    G = formGraph(T, pos, 1.6, seed=5)
#    n = G.number_of_nodes()
#    pos2 = dict(enumerate(pos))
#    fig, ax = showPaths(G, pos2, N)
#    plt.show()

    # Add a sink, connect all spareTSV to it.
    ## pos = pos + [(1.5,.5)]
    for u, v in G.edges():
        h = np.array(G.node[u]['pos']) - np.array(G.node[v]['pos'])
        G[u][v]['cost'] = np.sqrt(np.dot(h, h))
        #G[u][v]['cost'] = h[0] + h[1]

    for u, v in G.edges():
        G[u][v]['cost'] = np.log(abs(G[u][v]['cost']))

    cmax = max(c for _, _, c in G.edges.data('cost'))
    cmin = min(c for _, _, c in G.edges.data('cost'))
    # mid = (cmax + cmin)/2.
    x0 = np.array([cmax, cmin])
    t = cmax - cmin
    E = ell(5.*t, x0)
    P = optscaling_oracle(G)
    xb, fb, niter, flag, status = cutting_plane_dc(P, E, 1.1*t, 2000, 1e-6)

    fmt = '{:f} {} {} {}'
    print(np.exp(xb))
    print(fmt.format(fb, niter, flag, status))
    assert flag == 1
    assert niter == 40
