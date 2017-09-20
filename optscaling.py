import scipy.sparse as sp
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

N = 100
M1 = sp.rand(N, N, density=0.1)
A = M1 + M1.T  # make symmetric
D = nx.to_networkx_graph(A,create_using=nx.DiGraph())
nx.bellman_ford(G, source)