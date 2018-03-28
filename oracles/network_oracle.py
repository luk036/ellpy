# -*- coding: utf-8 -*-
import numpy as np


class network_oracle:
    def __init__(self, G, edge_fun):
        self.G = G
        self.edge_fun = edge_fun

    def __call__(self, x, t):
        return x, t
