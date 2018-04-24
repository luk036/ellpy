import time
import numpy as np
#import matplotlib.pyplot as plt
#import cvxpy as cvx
#from scipy.signal import remez, minimum_phase, freqz, group_delay
from ..cutting_plane import cutting_plane_dc
from ..ell import ell
from .spectral_fact import spectral_fact
# from problem import Problem
from .lowpass_oracle import lowpass_oracle


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
N = 32
wpass = 0.12*np.pi   # end of passband
wstop = 0.20*np.pi   # start of stopband
delta0_wpass = 0.125
delta0_wstop = 0.125
# maximum passband ripple in dB (+/- around 0 dB)
delta = 20*np.log10(1 + delta0_wpass)
delta2 = 20*np.log10(delta0_wstop)      # stopband attenuation desired in dB

# *********************************************************************
# optimization parameters
# *********************************************************************
# rule-of-thumb discretization (from Cheney's Approximation Theory)
m = 15*N
w = np.linspace(0, np.pi, m)  # omega

# A is the matrix used to compute the power spectrum
# A(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos(N*w)]
An = 2*np.cos(np.outer(w, np.arange(1, N)))
A = np.hstack((np.ones((m, 1)), An))

# passband 0 <= w <= w_pass
ind_p = np.nonzero(w <= wpass)[0]    # passband
Lp = 10**(-delta/20)
Up = 10**(+delta/20)
Ap = A[ind_p, :]

# stopband (w_stop <= w)
ind_s = np.nonzero(wstop <= w)[0]   # stopband
Sp = 10**(delta2/20)
As = A[ind_s, :]

# remove redundant contraints
# ind_nr = setdiff(1:m,ind_p)   # fullband less passband
# ind_nr = setdiff(ind_nr, ind_s) # luk: for making parallel cut
ind_nr = np.setdiff1d(np.arange(m), ind_p)
ind_nr = np.setdiff1d(ind_nr, ind_s)
Anr = A[ind_nr, :]

Lpsq = Lp**2
Upsq = Up**2
Spsq = Sp**2
# ********************************************************************
# optimization
# ********************************************************************

def test_lowpass():
    # tic = time.time()

    r0 = np.zeros(N)  # initial x0
    r0[0] = 0

    # Ae = diag(ones(N,1)) # initial ellipsoid (sphere)
    E = ell(4., r0)
    P = lowpass_oracle(Ap, As, Anr, Lpsq, Upsq)
    r, Spsq_new, num_iters, flag, status = cutting_plane_dc(
        P, E, Spsq, 1000, 1e-4)

    assert flag == 1
    # toc = time.time()

    # print(num_iters)

    # x = r
    # m = length(x)
    # u = x(m:-1:1)'
    # u(m) = 0.5*x(1)
    # d = roots(u)
    # figure(3)
    # plot(1./d,'x')
    # axis('square')
    # grid on
    # hold on
    # elplot([1 0 0 1], [0 0])

    #E = ell(1,r0)
    #P = FIR_oracle2(Ap, As, Anr, Lpsq, Upsq)
    # [r, Spsq_new, iter, flag, status] ...
    #  = ellipsoid_dc(@P.assess, E, Spsq, 100000, 1e-4)
    # toc
    # iter


    # *********************************************************************
    # plotting routines
    # *********************************************************************
    # frequency response of the designed filter, where j = sqrt(-1)
    h_sp = spectral_fact(r)      # from CVX distribution, Examples subdirectory
    h = h_sp
    # compute the min attenuation in the stopband (convert to original vars)
    Ustop = 20*np.log10(np.sqrt(Spsq_new))

    print('Min attenuation in the stopband is ', Ustop, ' dB.')

    # freq = [0, 0.12, 0.2, 1.0]
    # desired = [1, 0]
    # h_linear = remez(151, freq, desired, Hz=2.)
    # h_min_hom = minimum_phase(h_linear, method='homomorphic')

    # fig, axs = plt.subplots(4, figsize=(4, 8))
    # for h, style, color in zip((h_linear, h_min_hom, h_sp),
    #                            ('-', '-', '--'), ('k', 'r', 'c')):
    #     #if flag == 1:
    #     w, H = freqz(h)
    #     w, gd = group_delay((h, 1))
    #     w /= np.pi
    #     axs[0].plot(h, color=color, linestyle=style)
    #     axs[1].plot(w, np.abs(H), color=color, linestyle=style)
    #     axs[2].plot(w, 20 * np.log10(np.abs(H)), color=color, linestyle=style)
    #     axs[3].plot(w, gd, color=color, linestyle=style)

    # for ax in axs:
    #     ax.grid(True, color='0.5')
    #     ax.fill_between(freq[1:3], *ax.get_ylim(), color='    #ffeeaa', zorder=1)
    # axs[0].set(xlim=[0, len(h_linear) - 1], ylabel='Amplitude', xlabel='Samples')
    # axs[1].legend(['Linear', 'Min-Hom', 'Our'], title='Phase')
    # for ax, ylim in zip(axs[1:], ([0, 1.1], [-150, 10], [-60, 60])):
    #     ax.set(xlim=[0, 1], ylim=ylim, xlabel='Frequency')
    # axs[1].set(ylabel='Magnitude')
    # axs[2].set(ylabel='Magnitude (dB)')
    # axs[3].set(ylabel='Group delay')
    # plt.tight_layout()
    # plt.show()

    # H = [exp(-j*kron(w,[0:N-1]))]*h
    # figure(2)
    # subplot(121)
    #     # magnitude
    # plot(w,20*log10(abs(H)), ...
    #    [0 wpass],[delta delta],'r--', ...
    #    [0 wpass],[-delta -delta],'r--', ...
    #    [wstop pi],[Ustop Ustop],'r--')
    # xlabel('w')
    # ylabel('mag H(w) in dB')
    # axis([0 pi -50 5])
    # title(sprintf('N=    #d, w_p(pi)=#3.2f, w_s(pi)=#3.2f, delta=#3.2f', N, wpass/pi, wstop/pi, delta))

    # #compare impulse response designed by conventional method
    # subplot(122)
    # ## h_sp = spectral_fact(r)  #from CVX distribution, Examples subdirectory
    # plot([0:N-1],h_sp','+r--')
    # hold on
    # plot([0:N-1],h(end:-1:1)','ob:')
    # legend('conventional','optimal')
    # xlabel('t'), ylabel('h(t)') grid
    # title(sprintf('h_{max} conventional=    #3.4f, h_{max} optimal=#3.4f',max(abs(h_sp)),max(abs(h))))
    # set(gcf,'Outerposition',[300 300 256*4 256*2])

    # figure(1)
    #     # FIR impulse response
    # plot([0:N-1],h','ob:')
    # xlabel('t'), ylabel('h(t)')
