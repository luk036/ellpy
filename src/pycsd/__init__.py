"""
PyCSD
=====
"""

from __future__ import absolute_import

import sys
if sys.version_info[:2] < (2, 7):
    m = "Python 2.7 or later is required for PyCSD (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

# Release data
# from pycsd import release

# from pycsd.csd import *
