#! /usr/bin/env python
""" 
 Unittests for the CSD module
 
"""

import csd
import unittest

good_values_dict = { 32  : '+00000',
                     -32 : '-00000',
                     0   : '0',
                     7   : '+00-',
                     15  : '+000-'
                    }

class tests__integers( unittest.TestCase ):

    def test__01_to_integer(self):
        """ Check conversion from CSD to integer """
        
        for key in good_values_dict.keys():
            csd_str = good_values_dict[key] 
            value = csd.to_decimal(csd_str)
            self.assertEquals(value, key )

        
    def test__02_to_csd(self):
        """ Check that integers are converted to CSD properly. """
        
        for key in good_values_dict.keys():
            csd_str = csd.to_csd(key)
            self.assertEquals(csd_str, good_values_dict[key] )
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(tests__integers))
    return suite
        
if __name__ == '__main__':
    unittest.main()

