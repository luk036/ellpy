# -*- coding: utf-8 -*-
import numpy as np
from .lmi_oracle import *


class sdp_oracle:

    def __init__(self, c, F, B):
        self.c = c
        self.lmi = lmi_oracle(F, B)

    def __call__(self, x, t):
        f0 = np.dot(self.c, x)
        fj = f0 - t
        if fj > 0.:
            return self.c, fj, t

        g, fj = self.lmi.chk_spd(x)
        if fj > 0.:
            return g, fj, t
        return self.c, 0., f0
