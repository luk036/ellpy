# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
#import cvxpy as cvx
from ..cutting_plane import cutting_plane_dc
from ..ell import ell
from ..problem import Problem

# ********************************************************************
# Problem specs.
# ********************************************************************
# Number of FIR coefficients (including the zeroth one).
n = 20

# Rule-of-thumb frequency discretization (Cheney's Approx. Theory book).
m = 15*n
w = np.mat(np.linspace(0, np.pi, m)).T

# ********************************************************************
# Construct the desired filter.
# ********************************************************************
# Fractional delay.
D = 8.25                # Delay value.
Hdes = np.exp(-1j*D*w)  # Desired frequency response.

# Gaussian filter with linear phase. (Uncomment lines below for this design.)
#var = 0.05
#Hdes = 1/(np.sqrt(2*np.pi*var)) * np.exp(-np.square(w-np.pi/2)/(2*var))
#Hdes = np.multiply(Hdes, np.exp(-1j*n/2*w))

# A is the matrix used to compute the frequency response
# from a vector of filter coefficients:
#     A[w,:] = [1 exp(-j*w) exp(-j*2*w) ... exp(-j*n*w)]
A = np.exp(-1j * np.kron(np.mat(w), np.mat(np.arange(n))))

# Presently CVXPY does not do complex-valued math, so the
# problem must be formatted into a real-valued representation.

# Split Hdes into a real part, and an imaginary part.
Hdes_r = np.real(Hdes)
Hdes_i = np.imag(Hdes)

# Split A into a real part, and an imaginary part.
A_R = np.real(A)
A_I = np.imag(A)


# Optimal Chebyshev filter formulation.
class my_oracle:
    def __init__(self):
        self.A_R = np.squeeze(np.asarray(A_R))
        self.A_I = np.squeeze(np.asarray(A_I))
        self.Hdes_r = np.squeeze(np.asarray(Hdes_r))
        self.Hdes_i = np.squeeze(np.asarray(Hdes_i))
        self.m = m

    def __call__(self, h, t):
        fmax = float('-Inf')
        for i in range(self.m):
            a_R = self.A_R[i, :]
            a_I = self.A_I[i, :]
            H_r = self.Hdes_r[i]
            H_i = self.Hdes_i[i]
            t_r = a_R.dot(h) - H_r
            t_i = a_I.dot(h) - H_i
            fj = t_r**2 + t_i**2
            if fj >= t:
                g = 2. * (t_r * a_R + t_i*a_I).T
                return (g, fj - t), t
            if fmax < fj:
                fmax = fj
                imax = i
        t = fmax
        a_R = self.A_R[imax, :]
        a_I = self.A_I[imax, :]
        H_r = self.Hdes_r[imax]
        H_i = self.Hdes_i[imax]
        t_r = a_R.dot(h) - H_r
        t_i = a_I.dot(h) - H_i
        g = 2.*(t_r*a_R + t_i*a_I).T
        return (g, 0.), t


def test_firfilter():
    h0 = np.zeros(n)  # initial x0
    E = ell(10., h0)
    P = my_oracle()
    #hb, fb, niter, flag, status = cutting_plane_dc(P, E, 10., 2000, 1e-8)
    prob1 = Problem(E, P)
    prob1.solve(100.)

    print('Problem status:', prob1.status)

    print("optimal value", prob1.optim_value)
    assert prob1.status == 'optimal'

    #fmt = '{:f} {} {} {}'
    # print(prob1.optim_var)
    #print(fmt.format(prob1.optim_vale, prob1.solver_stats.num_iters))

    #print 'Problem status:', flag
    # if flag != 1:
    #    raise Exception('ELL Error')

    #hv = np.asmatrix(prob1.optim_var).T