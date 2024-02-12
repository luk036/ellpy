# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellpy.oracles.chol_ext import chol_ext


def test_chol1():
    """[summary]"""
    l1 = [[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
    m1 = np.array(l1)
    Q1 = chol_ext(len(m1))
    assert Q1.factorize(m1)


def test_chol2():
    """[summary]"""
    l2 = [
        [18.0, 22.0, 54.0, 42.0],
        [22.0, -70.0, 86.0, 62.0],
        [54.0, 86.0, -174.0, 134.0],
        [42.0, 62.0, 134.0, -106.0],
    ]
    m2 = np.array(l2)
    Q = chol_ext(len(m2))
    assert not Q.factorize(m2)
    Q.witness()
    assert Q.p == (0, 2)
    # assert ep == 1.0


def test_chol3():
    """[summary]"""
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
    m3 = np.array(l3)
    Q = chol_ext(len(m3))
    assert not Q.factorize(m3)
    ep = Q.witness()
    assert Q.p == (0, 1)
    assert Q.v[0] == 1.0
    assert ep == 0.0


def test_chol4():
    """[summary]"""
    l1 = [[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
    m1 = np.array(l1)
    Q1 = chol_ext(len(m1))
    Q1.allow_semidefinite = True
    assert Q1.factorize(m1)


def test_chol5():
    """[summary]"""
    l2 = [
        [18.0, 22.0, 54.0, 42.0],
        [22.0, -70.0, 86.0, 62.0],
        [54.0, 86.0, -174.0, 134.0],
        [42.0, 62.0, 134.0, -106.0],
    ]
    m2 = np.array(l2)
    Q = chol_ext(len(m2))
    Q.allow_semidefinite = True
    assert not Q.factorize(m2)
    Q.witness()
    assert Q.p == (0, 2)
    # assert ep == 1.0


def test_chol6():
    """[summary]"""
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]
    m3 = np.array(l3)
    Q = chol_ext(len(m3))
    Q.allow_semidefinite = True
    assert Q.factorize(m3)


#     [v, ep] = Q.witness2()
#     assert len(v) == 1
#     assert v[0] == 1.0
#     assert ep == 0.0


def test_chol7():
    """[summary]"""
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, -20.0]]
    m3 = np.array(l3)
    Q = chol_ext(len(m3))
    Q.allow_semidefinite = True
    assert not Q.factorize(m3)
    ep = Q.witness()
    assert ep == 20.0


def test_chol8():
    """[summary]"""
    """[summary]
    """
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 20.0]]
    m3 = np.array(l3)
    Q = chol_ext(len(m3))
    Q.allow_semidefinite = False
    assert not Q.factorize(m3)


def test_chol9():
    """[summary]"""
    """[summary]
    """
    l3 = [[0.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 20.0]]
    m3 = np.array(l3)
    Q = chol_ext(len(m3))
    Q.allow_semidefinite = True
    assert Q.factorize(m3)
