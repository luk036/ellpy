# -*- coding: utf-8 -*-
"""
Negative cycle detection for weighed graphs.
(based on Bellman-Ford algorithm)
"""

from collections import deque
# from heapq import heappush, heappop
# from itertools import count
import networkx as nx
# from networkx.utils import generate_unique_node


def find_neg_cycle(G, pred, dist, source, weight='weight'):
    """Compute negative cycle in weighted graphs.

    The algorithm has a running time of O(mn) where n is the number of
    nodes and m is the number of edges.  

    Parameters
    ----------
    G : NetworkX graph
       The algorithm works for all types of graphs, including directed
       graphs and multigraphs.

    source: node label
       Starting node for path

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    Returns
    -------
    pred, dist : dictionaries
       Returns two dictionaries keyed by node to predecessor in the
       negative cycle and to the distance from the source respectively.

    Raises
    ------
    NetworkXUnbounded
       If the (di)graph does not contain a negative cost (di)cycle, the
       algorithm raises an exception.  Note: any negative weight edge in an
       undirected graph is a negative cost cycle.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionaries returned only have keys for nodes reachable from
    the source.

    In the case where the (di)graph is not connected, if a component
    not containing the source contains a negative cost (di)cycle, it
    will not be detected.

    """
    if source not in G:
        raise KeyError("Node %s is not found in the graph" % source)

    for u, v, attr in G.selfloop_edges(data=True):
        if attr.get(weight, 1) < 0:
            raise nx.NetworkXUnbounded("Self loop negative cost cycle detected.")

    if len(G) == 1:
        return None

    v = _neg_cycle_relaxation(G, pred, dist, [source], weight)
    return v


def _neg_cycle_relaxation(G, pred, dist, source, weight):
    """Relaxation loop for Bellman–Ford algorithm

    Parameters
    ----------
    G : NetworkX graph

    pred: dict
        Keyed by node to predecessor in the path

    dist: dict
        Keyed by node to the distance from the source

    source: list
        List of source nodes

    weight: string
       Edge data key corresponding to the edge weight

    Returns
    -------
    Returns two dictionaries keyed by node to predecessor in the
       path and to the distance from the source respectively.

    Raises
    ------
    NetworkXUnbounded
       If the (di)graph contains a negative cost (di)cycle, the
       algorithm raises an exception to indicate the presence of the
       negative cost (di)cycle.  Note: any negative weight edge in an
       undirected graph is a negative cost cycle
    """
    if G.is_multigraph():
        def get_weight(edge_dict):
            return min(eattr.get(weight, 1) for eattr in edge_dict.values())
    else:
        def get_weight(edge_dict):
            return edge_dict.get(weight, 1)

    G_succ = G.succ if G.is_directed() else G.adj
    inf = float('inf')
    n = len(G)

    count = {}
    q = deque(source)
    in_q = set(source)
    while q:
        u = q.popleft()
        in_q.remove(u)
        # Skip relaxations if the predecessor of u is in the queue.
        if pred[u] not in in_q:
            dist_u = dist[u]
            for v, e in G_succ[u].items():
                dist_v = dist_u + get_weight(e)
                if dist_v < dist.get(v, inf):
                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0) + 1
                        if count_v == n:
                            return v
                        count[v] = count_v
                    dist[v] = dist_v
                    pred[v] = u

    return None


if __name__=="__main__":
    from networkx.utils import generate_unique_node

    G = nx.cycle_graph(5, create_using = nx.DiGraph())
    G[1][2]['weight'] = -4
    newnode = generate_unique_node()
    G.add_edges_from([(newnode, n) for n in G])

    dist = {newnode: 0}
    pred = {newnode: None}
    #source = 0
    #dist = {source: 0}
    #pred = {source: None}

    v = find_neg_cycle(G, pred, dist, newnode)
    print v
    print sorted(pred.items())
    print sorted(dist.items())

    source = 0
    dist = {source: 0}
    pred = {source: None}
    G = nx.path_graph(5, create_using = nx.DiGraph())
    v = find_neg_cycle(G, pred, dist, source)
    print v
    print sorted(pred.items())
    print sorted(dist.items())

