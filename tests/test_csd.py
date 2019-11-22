# -*- coding: utf-8 -*-
from pycsd.csd import to_csdfixed, to_decimal, to_csd


def test_csd1():
    csdstr = "+00-00+"
    csdnumber = to_decimal(csdstr)
    csdnew = to_csd(csdnumber)
    assert csdnew == csdstr


def test_csd2():
    csdstr = "+00-.000+"
    csdnumber = to_decimal(csdstr)
    csdnew = to_csd(csdnumber, places=4)
    assert csdnew == csdstr


def test_csd3():
    csdstr = "+00-.000+"
    csdnumber = to_decimal(csdstr)
    csdnew = to_csdfixed(csdnumber, nnz=3)
    assert csdnew == csdstr


def test_csd4():
    n = 545
    csdstr = to_csd(n)
    n2 = to_decimal(csdstr)
    assert n == n2
