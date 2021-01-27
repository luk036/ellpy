#! /usr/bin/env python3
"""
 Canonical Signed Digit Functions

 Handles:
  * Decimals
  *
  *

 eg, +00-00+000.0 or 0.+0000-00+
 Where: '+' is +1
        '-' is -1

 Harnesser
 License: GPL2
"""
from __future__ import print_function

from math import ceil, fabs, log


def to_csd(num, places=0, debug=False):
    """ Convert the argument to CSD Format. """

    if debug:
        print("Converting %f " % (num), )

    # figure out binary range, special case for 0
    if num == 0.:
        return '0'

    absnum = fabs(num)
    n = 0 if absnum < 1. else ceil(log(absnum * 1.5, 2))
    csd_str = '0' if absnum < 1. else ''

    if debug:
        print("to %d.%d format" % (n, places))

    limit = pow(2., n) / 3.
    while (n > -places):
        if debug:
            print("  ", num, limit)

        # decimal point?
        if n == 0:  # unlikely
            csd_str += '.'

        n -= 1
        # convert the number
        if num > limit:
            csd_str += '+'
            num -= 1.5 * limit
        elif num < -limit:
            csd_str += '-'
            num += 1.5 * limit
        else:
            csd_str += '0'
        limit /= 2.

        if debug:
            print(csd_str)

    return csd_str


def to_decimal(csd_str, debug=False):
    """ Convert the CSD string to a decimal """

    if debug:
        print("Converting: ", csd_str)

    num = 0.
    loc = 0
    for i, c in enumerate(csd_str):
        num *= 2.
        if c == '+':
            num += 1.
        elif c == '-':
            num -= 1.
        elif c == '0':
            pass
        elif c == '.':  # unlikely
            num /= 2.
            loc = i + 1
        else:
            raise ValueError
    if loc != 0:
        num /= pow(2., len(csd_str) - loc)
    return num


# def to_decimal_old(csd_str, debug=False):
#     """ Convert the CSD string to a decimal """

#     if debug:
#         print("Converting: ", csd_str)

#     #  Find out what the MSB power of two should be, keeping in
#     # mind we may have a fractional CSD number
#     try:
#         (m, n) = csd_str.split('.')
#         csd_str = csd_str.replace('.', '')  # get rid of point now...
#     except ValueError:
#         m = csd_str
#         n = ""

#     msb_power = len(m) - 1

#     num = 0.0
#     for ii, c in enumerate(csd_str):

#         power_of_two = 2.0**(msb_power - ii)

#         if c == '+':
#             num += power_of_two
#         elif c == '-':
#             num -= power_of_two

#         if debug:
#             print('  "%s" (%d.%d); 2**%d = %d; Num=%f' %
#                   (c, len(m), len(n), msb_power - ii, power_of_two, num))

#     return num


def to_csdfixed(num, nnz=4, debug=False):
    """ Convert the argument to CSD Format. """

    if debug:
        print("Converting %f " % (num), )

    # figure out binary range, special case for 0
    if num == 0.:
        return '0'
    absnum = fabs(num)
    n = 0 if absnum < 1. else ceil(log(absnum * 1.5, 2))
    csd_str = '0' if absnum < 1. else ''
    limit = pow(2., n) / 3.
    while (n > 0 or nnz > 0):
        if debug:
            print("  ", num, limit)

        # decimal point?
        if n == 0:
            csd_str += '.'

        n -= 1
        # convert the number
        if num > limit:
            csd_str += '+'
            num -= 1.5 * limit
            nnz -= 1
        elif num < -limit:
            csd_str += '-'
            num += 1.5 * limit
            nnz -= 1
        else:
            csd_str += '0'
        limit /= 2.

        if nnz == 0:
            num = 0

        if debug:
            print(csd_str)

    return csd_str
