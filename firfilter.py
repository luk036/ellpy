import numpy as np
import cvxpy as cvx
from cutting_plane import cutting_plane_dc
from ell import ell

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
                return g, fj - t, t
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
        return g, 0., t


h0 = np.zeros(n)  # initial x0
E = ell(10., h0)
P = my_oracle()
hb, fb, niter, flag, status = cutting_plane_dc(P, E, 10., 2000, 1e-8)

fmt = '{:f} {} {} {}'
print(hb)
print(fmt.format(fb, niter, flag, status))

print 'Problem status:', flag
if flag != 1:
    raise Exception('ELL Error')
hv = np.asmatrix(hb).T

# h is the (real) FIR coefficient vector, which we are solving for.
h = cvx.Variable(n)
# The objective is:
#     minimize max(|A*h-Hdes|)
# but modified into an equivelent form:
#     minimize max( real(A*h-Hdes)^2 + imag(A*h-Hdes)^2 )
# such that all computation is done in real quantities only.
obj = cvx.Minimize(
        cvx.max_entries(
           cvx.square(A_R * h - Hdes_r)     # Real part.
         + cvx.square(A_I * h - Hdes_i) ) ) # Imaginary part.

# Solve problem.
prob = cvx.Problem(obj)
prob.solve()
hv = h.value

# Check if problem was successfully solved.
print 'Problem status:', prob.status
if prob.status != cvx.OPTIMAL:
    raise Exception('CVXPY Error')
print "optimal value", prob.value


import matplotlib.pyplot as plt

# Show plot inline in ipython.
# %matplotlib inline

# Plot properties.
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 16}
# plt.rc('font', **font)

# Plot the FIR impulse reponse.
plt.figure(figsize=(6, 6))
plt.stem(range(n), hv)
plt.xlabel('n')
plt.ylabel('h(n)')
plt.title('FIR filter impulse response')
plt.show()

# Plot the frequency response.
H = np.exp(-1j * np.kron(w, np.mat(np.arange(n)))) * hv
plt.figure(figsize=(6, 6))
# Magnitude
plt.plot(np.array(w), 20 * np.log10(np.array(np.abs(H))),
         label='optimized')
plt.plot(np.array(w), 20 * np.log10(np.array(np.abs(Hdes))), '--',
         label='desired')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$|H(\omega)|$ in dB')
plt.title('FIR filter freq. response magnitude')
plt.show()

plt.xlim(0, np.pi)
plt.ylim(-30, 10)
plt.legend(loc='lower right')
# Phase
plt.figure(figsize=(6, 6))
plt.plot(np.array(w), np.angle(np.array(H)))
plt.xlim(0, np.pi)
plt.ylim(-np.pi, np.pi)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\angle H(\omega)$')
plt.title('FIR filter freq. response angle')
plt.show()
