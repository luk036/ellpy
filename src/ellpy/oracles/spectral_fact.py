import numexpr as ne
import numpy as np


def spectral_fact(r):
    """Spectral factorization using Kolmogorov 1939 approach.
      (code follows pp. 232-233, Signal Analysis, by A. Papoulis)

      Computes the minimum-phase impulse response which satisfies
      given auto-correlation.

      Input:
        r: top-half of the auto-correlation coefficients
           starts from 0th element to end of the auto-corelation
           should be passed in as a column vector
      Output
        h: impulse response that gives the desired auto-correlation
    """

    # length of the impulse response sequence
    n = len(r)

    # over-sampling factor
    mult_factor = 100  # should have mult_factor*(n) >> n
    m = mult_factor * n

    # computation method:
    # H(exp(jTw)) = alpha(w) + j*phi(w)
    # where alpha(w) = 1/2*ln(R(w)) and phi(w) = Hilbert_trans(alpha(w))

    # compute 1/2*ln(R(w))
    # w = 2*pi*[0:m-1]/m
    w = np.linspace(0, 2 * np.pi, m, endpoint=False)
    # R = [ones(m, 1) 2*cos(kron(w', [1:n-1]))]*r
    Bn = np.outer(w, np.arange(1, n))
    An = 2 * np.cos(Bn)
    R = np.hstack((np.ones((m, 1)), An)) @ r  # NOQA

    alpha = ne.evaluate('0.5 * log(abs(R))')

    # find the Hilbert transform
    alphatmp = np.fft.fft(alpha)
    # alphatmp(floor(m/2)+1: m) = -alphatmp(floor(m/2)+1: m)
    ind = int(m / 2)  # python3 need int()
    alphatmp[ind:m] = -alphatmp[ind:m]
    alphatmp[0] = 0
    alphatmp[ind] = 0
    phi = np.real(np.fft.ifft(1j * alphatmp))

    # now retrieve the original sampling
    # index = find(np.reminder([0:m-1], mult_factor) == 0)
    index = np.arange(m, step=mult_factor)
    alpha1 = alpha[index]
    phi1 = phi[index]

    # compute the impulse response (inverse Fourier transform)
    h = np.real(np.fft.ifft(np.exp(alpha1 + 1j * phi1), n))

    return h


def inverse_spectral_fact(h):
    """[summary]

    Arguments:
        h ([type]): [description]

    Returns:
        [type]: [description]
    """
    n = len(h)
    r = np.zeros(n)
    for t in range(n):
        r[t] = h[t:] @ h[:n - t]
    return r


# if __name__ == "__main__":
#     r = np.random.rand(20)
#     h = spectral_fact(r)
#     print(h)
