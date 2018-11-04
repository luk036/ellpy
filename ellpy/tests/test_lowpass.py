# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import numpy as np
from ellpy.cutting_plane import cutting_plane_dc, cutting_plane_q, Options
from ellpy.ell import ell
from .lowpass_oracle import lowpass_oracle
from .csdlowpass_oracle import csdlowpass_oracle


# Modified from CVX code by Almir Mutapcic in 2006.
# Adapted in 2010 for impulse response peak-minimization by convex iteration by Christine Law.
#
# "FIR Filter Design via Spectral Factorization and Convex Optimization"
# by S.-P. Wu, S. Boyd, and L. Vandenberghe
#
# Designs an FIR lowpass filter using spectral factorization method with
# constraint on maximum passband ripple and stopband attenuation:
#
#   minimize   max |H(w)|                      for w in stopband
#       s.t.   1/delta <= |H(w)| <= delta      for w in passband
#
# We change variables via spectral factorization method and get:
#
#   minimize   max R(w)                          for w in stopband
#       s.t.   (1/delta)**2 <= R(w) <= delta**2  for w in passband
#              R(w) >= 0                         for all w
#
# where R(w) is squared magnitude frequency response
# (and Fourier transform of autocorrelation coefficients r).
# Variables are coeffients r and G = hh' where h is impulse response.
# delta is allowed passband ripple.
# This is a convex problem (can be formulated as an SDP after sampling).

# rand('twister',sum(100*clock))
# randn('state',sum(100*clock))

# *********************************************************************
# filter specs (for a low-pass filter)
# *********************************************************************
# number of FIR coefficients (including zeroth)

# *********************************************************************
# filter specs (for a low-pass filter)
# *********************************************************************
# number of FIR coefficients (including zeroth)
def create_lowpass_case(N=48):
    wpass = 0.12*np.pi   # end of passband
    wstop = 0.20*np.pi   # start of stopband
    delta0_wpass = 0.125
    delta0_wstop = 0.125
    # maximum passband ripple in dB (+/- around 0 dB)
    delta = 20*np.log10(1 + delta0_wpass)
    # stopband attenuation desired in dB
    delta2 = 20*np.log10(delta0_wstop)

    # *********************************************************************
    # optimization parameters
    # *********************************************************************
    # rule-of-thumb discretization (from Cheney's Approximation Theory)
    m = 15*N
    w = np.linspace(0, np.pi, m)  # omega

    # A is the matrix used to compute the power spectrum
    # A(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos(N*w)]
    An = 2*np.cos(np.outer(w, np.arange(1, N)))
    A = np.concatenate((np.ones((m, 1)), An), axis=1)

    # passband 0 <= w <= w_pass
    ind_p = np.where(w <= wpass)[0]    # passband
    Lp = 10**(-delta/20)
    Up = 10**(+delta/20)
    Ap = A[ind_p, :]

    # stopband (w_stop <= w)
    ind_s = np.where(wstop <= w)[0]   # stopband
    Sp = 10**(delta2/20)
    As = A[ind_s, :]

    # remove redundant contraints
    # ind_nr = setdiff(1:m,ind_p)   # fullband less passband
    # ind_nr = setdiff(ind_nr, ind_s) # luk: for making parallel cut
    # ind_nr = np.setdiff1d(np.arange(m), ind_p)
    # ind_nr = np.setdiff1d(ind_nr, ind_s)
    # Anr = A[ind_nr, :]
    ind_beg = ind_p[-1]
    ind_end = ind_s[0]
    Anr = A[range(ind_beg+1, ind_end), :]

    Lpsq = Lp**2
    Upsq = Up**2
    Spsq = Sp**2

    P = lowpass_oracle(Ap, As, Anr, Lpsq, Upsq)
    return P, Spsq


def create_csdlowpass_case(N=48, nnz=8):
    P, Spsq = create_lowpass_case(N)
    Pcsd = csdlowpass_oracle(nnz, P)
    return Pcsd, Spsq


def run_lowpass(use_parallel_cut, duration=0.000001):
    N = 32

    r0 = np.zeros(N)  # initial x0
    r0[0] = 0
    E = ell(4., r0)
    E.use_parallel_cut = use_parallel_cut
    P, Spsq = create_lowpass_case(N)
    options = Options()
    options.max_it = 20000
    options.tol = 1e-8
    _, _, num_iters, feasible, _ = cutting_plane_dc(
        P, E, Spsq, options)
    time.sleep(duration)
    # h = spectral_fact(r)
    return num_iters, feasible


# def test_lowpass0(benchmark):
#     result = benchmark(run_lowpass, False)
#     assert result == 13442

# def test_lowpass1(benchmark):
#     result = benchmark(run_lowpass, True)
#     assert result == 568

def test_lowpass():
    result, feasible = run_lowpass(True)
    assert feasible
    assert result > 400
    assert result < 530


def run_csdlowpass(use_parallel_cut, duration=0.000001):
    N = 32
    nnz = 8

    r0 = np.zeros(N)  # initial x0
    r0[0] = 0
    E = ell(4., r0)
    E.use_parallel_cut = use_parallel_cut
    Pcsd, Spsq = create_csdlowpass_case(N, nnz)
    options = Options()
    options.max_it = 20000
    options.tol = 1e-8
    _, _, num_iters, feasible, _ = cutting_plane_q(
        Pcsd, E, Spsq, options)
    time.sleep(duration)
    # h = spectral_fact(r)
    return num_iters, feasible


def test_csdlowpass():
    result, feasible = run_csdlowpass(True)
    assert feasible
    assert result > 400
    assert result < 530
