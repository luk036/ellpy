#! /usr/bin/env python
""" 
 Unittests for the CSD module
 
"""

import sys
sys.path.append('../csd')
import csd
print dir(csd)
import unittest

good_values_dict = { 32.5 : '+0000.+' } 
class test__integer_conversion( unittest.TestCase ):

    def testToCSD(self):
        """ Check that integers are converted to CSD properly. """
        
        for key in good_values_dict.keys():
            csd_str = csd.to_csd(key)
            self.assert_(csd_str == good_values_dict[key] )
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(IntegerConversion))
    return suite

        
if __name__ == '__main__':
    unittest.main()

