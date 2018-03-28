from __future__ import print_function 
from pprint import pprint
import networkx as nx
from networkx.utils import generate_unique_node

def find_neg_cycle(G, weight='weight'):
    """Return True if there exists a negative edge cycle anywhere in G.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.

       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.

    Returns
    -------
    negative_cycle : bool
        True if a negative edge cycle exists, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.cycle_graph(5, create_using = nx.DiGraph())
    >>> print(nx.negative_edge_cycle(G))
    False
    >>> G[1][2]['weight'] = -7
    >>> print(nx.negative_edge_cycle(G))
    True

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    This algorithm uses bellman_ford_predecessor_and_distance() but finds
    negative cycles on any component by first adding a new node connected to
    every node, and starting bellman_ford_predecessor_and_distance on that
    node.  It then removes that extra node.
    """
    newnode = generate_unique_node()
    G.add_edges_from([(newnode, n) for n in G])

    dist = {newnode: 0}
    pred = {newnode: [None]}

    try:
        pred, dist = nx.bellman_ford_predecessor_and_distance(G, newnode, weight)
    except nx.NetworkXUnbounded:
        return (True, pred, dist)
    finally:
        G.remove_node(newnode)
    return (False, pred, dist)


G = nx.cycle_graph(5, create_using = nx.DiGraph())
found1, pred1, dist1 = find_neg_cycle(G)
print(found1)
print(sorted(pred1.items()))
print(sorted(dist1.items()))
G[1][2]['weight'] = -7
found2, pred2, dist2 = find_neg_cycle(G)
print(found2)
print(sorted(pred2.items()))
print(sorted(dist2.items()))
