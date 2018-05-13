#! /usr/bin/env python
""" 
 Unittests for the CSD module
 
"""

import csd
import unittest

good_values_dict = { 16.5    : '+0000.+',
                     -16.5   : '-0000.-',
                     -2.5    : '-0.-',
                     0.5     : '0.+',
                     -0.5    : '0.-'
                     }

class tests__decimals( unittest.TestCase ):

    def tmp(self):
        pass


    def test__01_to_decimal(self):
        """ Check the conversion from CSD with a binary point to decimal. """

        for key in good_values_dict.keys():
            csd_str = good_values_dict[key] 
            value = csd.to_decimal(csd_str)
            self.assertEquals(value, key )

    def test__01a_to_decimal(self):
        self.assertEquals( csd.to_decimal('0.+00-'), 0.5 - (2.0 ** -4 ) )
        self.assertEquals( csd.to_decimal('0.0+0-'), 0.25 - (2.0 ** -4 ) )
        self.assertEquals( csd.to_decimal('0.+0-0'), 0.5 - (2.0 ** -3 ) )

        
    def test__02_to_csd_1_place(self):
        """ Check that decimals are converted to CSD properly. """

        for key in good_values_dict.keys():
            csd_str = csd.to_csd(key, places=1)
            self.assertEquals(csd_str, good_values_dict[key] )
        
            
    def test__03_to_csd_4_places(self):
        """ To four places """
        self.assertEquals( csd.to_csd(  0.0625, 4 ), '0.000+' )
        self.assertEquals( csd.to_csd( -0.0625, 4 ), '0.000-' )
        self.assertEquals( csd.to_csd(  0.25,   4 ), '0.0+00' )
        self.assertEquals( csd.to_csd( -0.25,   4 ), '0.0-00' )


    def test__04_to_csd_x_places(self):
        """ To four places """
        self.assertEquals( csd.to_csd(  0.0625, 4 ), '0.000+' )
        self.assertEquals( csd.to_csd( -0.0625, 4 ), '0.000-' )
        self.assertEquals( csd.to_csd(  0.25,   4 ), '0.0+00' )
        self.assertEquals( csd.to_csd( -0.25,   4 ), '0.0-00' )


        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(tests__decimals))
    return suite
        
if __name__ == '__main__':
    unittest.main()

