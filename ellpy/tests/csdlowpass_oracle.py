# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from .lowpass_oracle import lowpass_oracle
from .spectral_fact import spectral_fact, inverse_spectral_fact
from pycsd.csd import to_csdfixed, to_decimal


class csdlowpass_oracle:

    def __init__(self, nnz, Ap, As, Anr, Lpsq, Upsq):
        self.nnz = nnz
        self.lowpass = lowpass_oracle(Ap, As, Anr, Lpsq, Upsq)

    def __call__(self, r, Spsq, retry):
        cut, Spsq2 = self.lowpass(r, Spsq)
        if Spsq == Spsq2:  # infeasible
            return cut, r, Spsq2, 0

        h = spectral_fact(r)
        hcsd = np.array([to_decimal(to_csdfixed(hi, self.nnz)) for hi in h])
        rcsd = inverse_spectral_fact(hcsd)
        cut, Spsq2 = self.lowpass(rcsd, Spsq)
        return cut, rcsd, Spsq2, 1
