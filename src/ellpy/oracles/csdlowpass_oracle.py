# -*- coding: utf-8 -*-
from __future__ import print_function

from typing import Tuple

import numpy as np
from pycsd.csd import to_csdfixed, to_decimal

from .spectral_fact import inverse_spectral_fact, spectral_fact

# np.ndarray = np.ndarray
Cut = Tuple[np.ndarray, float]


class csdlowpass_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, nnz, lowpass):
        """[summary]

        Arguments:
            nnz {[type]} -- [description]
            lowpass {[type]} -- [description]
        """
        self.nnz = nnz
        self.lowpass = lowpass

    def __call__(self, r, Spsq, retry):
        """[summary]

        Arguments:
            r {[type]} -- [description]
            Spsq {[type]} -- [description]
            retry {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        cut, Spsq2 = self.lowpass(r, Spsq)
        if Spsq == Spsq2:  # infeasible
            return cut, r, Spsq2, 0

        h = spectral_fact(r)
        hcsd = np.array([to_decimal(to_csdfixed(hi, self.nnz)) for hi in h])
        rcsd = inverse_spectral_fact(hcsd)
        (gc, hc), Spsq2 = self.lowpass(rcsd, Spsq)
        hc += gc.dot(rcsd - r)
        return (gc, hc), rcsd, Spsq2, 1
