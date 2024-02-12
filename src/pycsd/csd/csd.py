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


def to_csd(dec_val, places=0, debug=False):
    """
    Convert the argument `dec_val` to a string in CSD Format.
    Parameters
    ----------
    dec_val : scalar (integer or real)
              decimal value to be converted to CSD format
    places: integer
        number of fractional places. Default is places = 0 (integer number)
    Returns
    -------
    string
        containing the CSD value
    Original author: Harnesser
    https://sourceforge.net/projects/pycsd/
    License: GPL2
    """

    if debug:
        print(
            "Converting %f " % (dec_val),
        )

    # figure out binary range, special case for 0
    if dec_val == 0:
        return "0"

    absnum = fabs(dec_val)
    n = 0 if absnum < 1.0 else ceil(log(absnum * 1.5, 2))
    csd_str = "0" if absnum < 1.0 else ""

    if debug:
        print("to %d.%d format" % (n, places))

    # limit = pow(2., n) / 3.
    pow2n = pow(2.0, n - 1)
    while n > -places:
        if debug:
            print("  ", dec_val, 2 * pow2n / 3)

        # decimal point?
        if n == 0:  # unlikely
            csd_str += "."

        n -= 1
        # convert the number
        d = 1.5 * dec_val
        if d > pow2n:
            csd_str += "+"
            dec_val -= pow2n
        elif d < -pow2n:
            csd_str += "-"
            dec_val += pow2n
        else:
            csd_str += "0"
        pow2n /= 2

        if debug:
            print(csd_str)

    return csd_str


def to_decimal(csd_str, debug=False):
    """Convert the CSD string to a decimal"""

    if debug:
        print("Converting: ", csd_str)

    dec_val = 0.0
    loc = 0
    for i, c in enumerate(csd_str):
        dec_val *= 2.0
        if c == "+":
            dec_val += 1.0
        elif c == "-":
            dec_val -= 1.0
        elif c == "0":
            pass
        elif c == ".":  # unlikely
            dec_val /= 2.0
            loc = i + 1
        else:
            raise ValueError
    if loc != 0:
        dec_val /= pow(2.0, len(csd_str) - loc)
    return dec_val


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

#     dec_val = 0.0
#     for ii, c in enumerate(csd_str):

#         power_of_two = 2.0**(msb_power - ii)

#         if c == '+':
#             dec_val += power_of_two
#         elif c == '-':
#             dec_val -= power_of_two

#         if debug:
#             print('  "%s" (%d.%d); 2**%d = %d; dec_val=%f' %
#                   (c, len(m), len(n), msb_power - ii, power_of_two, dec_val))

#     return dec_val


def to_csdfixed(dec_val, nnz=4, debug=False):
    """Convert the argument to CSD Format."""

    if debug:
        print(
            "Converting %f " % (dec_val),
        )

    # figure out binary range, special case for 0
    if dec_val == 0.0:
        return "0"
    absnum = fabs(dec_val)
    n = 0 if absnum < 1.0 else ceil(log(absnum * 1.5, 2))
    csd_str = "0" if absnum < 1.0 else ""
    # limit = pow(2., n) / 3.
    pow2n = pow(2.0, n - 1)
    while n > 0 or (nnz > 0 and fabs(dec_val) > 1e-100):
        if debug:
            print("  ", dec_val, 2 * pow2n / 3)

        # decimal point?
        if n == 0:
            csd_str += "."

        n -= 1
        # convert the number
        d = 1.5 * dec_val
        if d > pow2n:
            csd_str += "+"
            dec_val -= pow2n
            nnz -= 1
        elif d < -pow2n:
            csd_str += "-"
            dec_val += pow2n
            nnz -= 1
        else:
            csd_str += "0"
        pow2n /= 2.0

        if nnz == 0:
            dec_val = 0

        if debug:
            print(csd_str)

    return csd_str
