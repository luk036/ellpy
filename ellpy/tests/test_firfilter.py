# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import numpy as np
# import cvxpy as cvx
from ellpy.ell import ell
from ellpy.problem import Problem

# ********************************************************************
# Problem specs.
# ********************************************************************
# Number of FIR coefficients (including the zeroth one).
n = 32

# Rule-of-thumb frequency discretization (Cheney's Approx. Theory book).
m = 15*n
w = np.linspace(0, np.pi, m)

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
A = np.exp(-1j * np.outer(w, np.arange(n)))

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
    def __call__(self, h, t):
        """[summary]

        Arguments:
            h {[type]} -- [description]
            t {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        fmax = float('-Inf')
        for i in range(m):
            a_R = A_R[i, :]
            a_I = A_I[i, :]
            H_r = Hdes_r[i]
            H_i = Hdes_i[i]
            t_r = a_R.dot(h) - H_r
            t_i = a_I.dot(h) - H_i
            fj = t_r**2 + t_i**2
            if fj >= t:
                g = 2 * (t_r * a_R + t_i*a_I)
                return (g, fj - t), t
            if fmax < fj:
                fmax = fj
                gmax = 2 * (t_r * a_R + t_i*a_I)
        return (gmax, 0.), fmax


def run_firfilter(no_trick, duration=0.000001):
    """[summary]

    Arguments:
        no_trick {[type]} -- [description]

    Keyword Arguments:
        duration {float} -- [description] (default: {0.000001})

    Raises:
        Exception -- [description]
    """
    h0 = np.zeros(n)  # initial x0
    E = ell(10., h0)
    E._no_defer_trick = no_trick
    P = my_oracle()
    prob1 = Problem(E, P)
    prob1.solve(100.)
    time.sleep(duration)

    print('Problem status:', prob1.status)
    if prob1.status != 'optimal':
        raise Exception('ELL Error')
    print("optimal value", prob1.optim_value)
    assert prob1.status == 'optimal'
    return prob1._solver_stats.num_iters


def test_firfilter_no_trick(benchmark):
    """[summary]

    Arguments:
        benchmark {[type]} -- [description]
    """
    num_iters = benchmark(run_firfilter, True)
    assert num_iters <= 208


def test_firfilter_use_trick(benchmark):
    """[summary]

    Arguments:
        benchmark {[type]} -- [description]
    """
    num_iters = benchmark(run_firfilter, False)
    assert num_iters <= 208
