# -*- coding: utf-8 -*-
from __future__ import print_function
 
from pprint import pprint
from ell import *
from cutting_plane import *
import numpy as np
import ..lsq_corr_oracle import *

def lsq_corr_fn(Y, s)
    '''
    % 1. Assume that the spatial correlation is isotopic: 
    %   i.e. Sig_{ij} = rho(|| s_j - s_i ||)
    % 2. Assume $Y$ is positive semidefinite
    % Use matrix norm estimation.
    '''
    nn = 5; # 4 terms => cubic polynomial
    % x = [t, kappa, p(a)]
    x0 = np.ones(nn+1) # initial x0
    E = ell(100., x0)
    P = lsq_corr_oracle(Y, s, nn)
    u = 100.
    l = 0.
    bx = np.zeros(nn+1)
    tol = 1e-4
    while u - l > tol:
        t = (u + l)/2.0
        E = ell(100., x0)
        x, bf, iter, flag, status = cutting_plane_fea(P, E, t, 1000, tol)
        if flag == 1:
            u = t
            bx = x
        else:
            l = t

    kappa = bx[-1]
    a = bx[:-2]

