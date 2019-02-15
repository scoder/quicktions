import os
import unittest

import quicktions
F = quicktions.Fraction
gcd = quicktions._gcd

class CImportTest(unittest.TestCase):

    def setUp(self):
        self.build_test_module()

    def tearDown(self):
        self.remove_test_module()

    def build_test_module(self):
        self.module_code = '\n'.join([
            '# cython: language_level=3str',
            'from quicktions cimport Fraction',
            'def get_fraction():',
            '    return Fraction(1, 2)',
        ])
        self.base_path = os.path.abspath(os.path.dirname(__file__))
        self.module_name = 'quicktions_importtest'
        self.module_filename = os.path.join(self.base_path, '.'.join([self.module_name, 'pyx']))
        with open(self.module_filename, 'w') as f:
            f.write(self.module_code)

    def remove_test_module(self):
        for fn in os.listdir(self.base_path):
            if not fn.startswith(self.module_name):
                continue
            os.remove(os.path.join(self.base_path, fn))

    def test_cimport(self):
        self.build_test_module()
        import pyximport
        self.py_importer, self.pyx_importer = pyximport.install(inplace=True, language_level=3)

        from quicktions_importtest import get_fraction

        self.assertEqual(get_fraction(), F(1,2))

        pyximport.uninstall(self.py_importer, self.pyx_importer)



def test_main():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CImportTest))
    return suite

def main():
    suite = test_main()
    runner = unittest.TextTestRunner(sys.stdout, verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    main()
