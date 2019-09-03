# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellpy.oracles.chol_ext import chol_ext


def print_case(l1):
    """[summary]

    Arguments:
          l1 {[type]} -- [description]
    """
    m1 = np.array(l1)
    Q = chol_ext(len(m1))
    Q.factorize(m1)


def test_chol1():
    """[summary]
    """
    l1 = [[25., 15., -5.], [15., 18., 0.], [-5., 0., 11.]]
    m1 = np.array(l1)
    Q1 = chol_ext(len(m1))
    Q1.factorize(m1)
    assert Q1.is_spd()


def test_chol2():
    """[summary]
    """
    l2 = [[18., 22., 54., 42.], [22., -70., 86., 62.], [54., 86., -174., 134.],
          [42., 62., 134., -106.]]
    m2 = np.array(l2)
    Q = chol_ext(len(m2))
    Q.factorize(m2)
    assert not Q.is_spd()
    Q.witness()
    assert Q.p == (0, 2)
    # assert ep == 1.0


def test_chol3():
    """[summary]
    """
    l3 = [[0., 15., -5.], [15., 18., 0.], [-5., 0., 11.]]
    m3 = np.array(l3)
    Q = chol_ext(len(m3))
    Q.factorize(m3)
    assert not Q.is_spd()
    ep = Q.witness()
    assert Q.p == (0, 1)
    assert Q.v[0] == 1.0
    assert ep == 0.0


def test_chol4():
    """[summary]
    """
    l1 = [[25., 15., -5.], [15., 18., 0.], [-5., 0., 11.]]
    m1 = np.array(l1)
    Q1 = chol_ext(len(m1))
    Q1.allow_semidefinite = True
    Q1.factorize(m1)
    assert Q1.is_spd()


def test_chol5():
    """[summary]
    """
    l2 = [[18., 22., 54., 42.], [22., -70., 86., 62.], [54., 86., -174., 134.],
          [42., 62., 134., -106.]]
    m2 = np.array(l2)
    Q = chol_ext(len(m2))
    Q.allow_semidefinite = True
    Q.factorize(m2)
    assert not Q.is_spd()
    Q.witness()
    assert Q.p == (0, 2)
    # assert ep == 1.0


def test_chol6():
    """[summary]
    """
    l3 = [[0., 15., -5.], [15., 18., 0.], [-5., 0., 11.]]
    m3 = np.array(l3)
    Q = chol_ext(len(m3))
    Q.allow_semidefinite = True
    Q.factorize(m3)
    assert Q.is_spd()


#     [v, ep] = Q.witness2()
#     assert len(v) == 1
#     assert v[0] == 1.0
#     assert ep == 0.0


def test_chol7():
    """[summary]
    """
    l3 = [[0., 15., -5.], [15., 18., 0.], [-5., 0., -20.]]
    m3 = np.array(l3)
    Q = chol_ext(len(m3))
    Q.allow_semidefinite = True
    Q.factorize(m3)
    assert not Q.is_spd()
    ep = Q.witness()
    #     assert len(v) == 3
    #     assert Q.v[0] == 0.
    assert ep == 20.
