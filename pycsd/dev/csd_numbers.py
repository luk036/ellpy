#! /usr/bin/env python
""" Convert some CSD numbers to get a feel for the range. """

import sys
sys.path.append('/home/mmcgrana/lib/python/csd')
import csd

num_digits = 4

# recursion
csd_num = [ '0' ] * num_digits

indent_ = 2

csd_dict = {}

def is_proper_csd(csd_num):
    """ Check we've a valid CSD number
    Returns false if the array contains consequitive non-zero terms.
    """

    for i in range( 1, len(csd_num) ):
        if csd_num[i] != '0' and csd_num[i-1] != '0':
            return False

    return True


def cycle_bits( i ) :

    global indent_
    
    indent_ += 2
    
    for ch in ( '0', '-', '+' ):
        csd_num[i] = ch

        if( i > 0 ):
            cycle_bits( i-1 )
        else :
            if is_proper_csd(csd_num):
                csd_str = ''.join(csd_num)
                num = csd.to_decimal( csd_str )
                csd_dict[num] = csd_str

    indent_ -= 2


def wiki_table():
    """ A table of CSD numbers for the wiki. """

    ordered_keys = csd_dict.keys()
    ordered_keys.sort()

    wiki_str = """
{| BORDER=1
! CSD
! Decimal
|-
"""
    
    for key in ordered_keys:
        wiki_str += """
| <tt>%s</tt>
| %d
|- """ % ( csd_dict[key], key )

    wiki_str += "|}"
    return wiki_str

def show_csd():
    """ """

    ordered_keys = csd_dict.keys()
    ordered_keys.sort()
    
    for key in ordered_keys:
        print csd_dict[key], key


if __name__ == '__main__' :
    
    cycle_bits( num_digits-1 )

    print wiki_table()
    
    
