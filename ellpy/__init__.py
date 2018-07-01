"""
EllPy
=====
"""

from __future__ import absolute_import

import sys
if sys.version_info[:2] < (2, 7):
    m = "Python 2.7 or later is required for NetworkX (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

# Release data
from ellpy import release

# from ellpy.oracles import *
from ellpy.cutting_plane import *
from ellpy.ell import *
from ellpy.problem import Problem

import ellpy.oracles
from ellpy.oracles import *
# from ellpy.lsq_corr_ell import lsq_corr_poly, lsq_corr_bspline
