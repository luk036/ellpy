# -*- coding: utf-8 -*-
from __future__ import print_function

from typing import Tuple, Union

import numpy as np
from pycsd.csd import to_csdfixed, to_decimal

from .spectral_fact import inverse_spectral_fact, spectral_fact

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class csdlowpass_oracle:
    """[summary]

    Returns:
        [type]: [description]
    """
    rcsd = None

    def __init__(self, nnz, lowpass):
        """[summary]

        Arguments:
            nnz ([type]): [description]
            lowpass ([type]): [description]
        """
        self.nnz = nnz
        self.lowpass = lowpass

    def __call__(self, r: Arr, Spsq, retry: bool):
        """[summary]

        Arguments:
            r (Arr): [description]
            Spsq ([type]): [description]
            retry (int): [description]

        Returns:
            [type]: [description]
        """
        if not retry:  # retry due to no effect in the previous cut
            self.lowpass.retry = False
            cut, Spsq2 = self.lowpass(r, Spsq)
            if Spsq == Spsq2:  # infeasible
                return cut, r, Spsq2, True
            h = spectral_fact(r)
            hcsd = np.array([to_decimal(to_csdfixed(hi, self.nnz)) for hi in h])
            self.rcsd = inverse_spectral_fact(hcsd)

        (gc, hc), Spsq2 = self.lowpass(self.rcsd, Spsq)
        # no more alternative cuts?
        hc += gc @ (self.rcsd - r)
        more_alt = self.lowpass.more_alt and not retry
        return (gc, hc), self.rcsd, Spsq2, more_alt
