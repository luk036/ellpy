# -*- coding: utf-8 -*-
from __future__ import print_function
from pprint import pprint
import numpy as np

class oracle:
    def __init__(self, i):
        self.i2 = i

    def __call__(self):
        return self.i2


class max_oracle:
    def __init__(self, it):
        self.it = it

    def __call__(self):
        Lt = []
        Li = []
        while True:  
            try:  
                oracle = next(self.it)
                Lt += [oracle()]
            except StopIteration:  
                # 遇到StopIteration就退出循环  
                break
        Lt = np.array(Lt)
        t = np.max(Lt)
        ti = np.argmax(Lt)
        return t, ti

if __name__ == "__main__":
    k = 0
    m = 5
    M = max_oracle(oracle(i) for i in range(k, m)+range(k))
    t, ti = M()
    print(t, ti)