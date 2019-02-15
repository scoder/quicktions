import os
import glob
import unittest

from Cython.Build import Cythonize

import quicktions
F = quicktions.Fraction
gcd = quicktions._gcd

class CImportTest(unittest.TestCase):

    def setUp(self):
        self.module_files = []

    def tearDown(self):
        for fn in self.module_files:
            if os.path.exists(fn):
                os.remove(fn)

    def build_test_module(self):
        module_code = '\n'.join([
            '# cython: language_level=3str',
            'from quicktions cimport Fraction',
            'def get_fraction():',
            '    return Fraction(1, 2)',
        ])
        base_path = os.path.abspath(os.path.dirname(__file__))
        module_name = 'quicktions_importtest'
        module_filename = os.path.join(base_path, '.'.join([module_name, 'pyx']))
        with open(module_filename, 'w') as f:
            f.write(module_code)

        Cythonize.main(['-i', module_filename])

        for fn in glob.glob(os.path.join(base_path, '.'.join([module_name, '*']))):
            self.module_files.append(os.path.abspath(fn))

    def test_cimport(self):
        self.build_test_module()

        from quicktions_importtest import get_fraction

        self.assertEqual(get_fraction(), F(1,2))


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
