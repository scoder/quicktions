
import os
import sys
import re

try:
    import setuptools
except ImportError:
    pass

from distutils.core import setup, Extension

try:
    from Cython.Build import cythonize
    import Cython.Compiler.Options as cython_options
    cython_options.annotate = True
    cython_available = True
except ImportError:
    cython_available = False
    cython = None

try:
    sys.argv.remove("--with-profile")
except ValueError:
    enable_profiling = False
else:
    enable_profiling = True

ext_modules = None
try:
    sys.argv.remove("--with-cython")
except ValueError:
    cythonize = None
else:
    if cython_available:
        compiler_directives = {}
        if enable_profiling:
            compiler_directives['profile'] = True
        ext_modules = cythonize('quicktions/*.pyx', compiler_directives=compiler_directives)
if ext_modules is None:
    ext_modules = [
        Extension("quicktions", [os.path.join("quicktions", "quicktions.c")]),
    ]

with open('quicktions/quicktions.pyx') as f:
    version = re.search("__version__\s*=\s*'([^']+)'", f.read(2048)).group(1)

with open('README.rst') as f:
    long_description = ''.join(f.readlines()[3:]).strip()

with open('CHANGES.rst') as f:
    long_description += '\n\n' + f.read()


setup(
    name="quicktions",
    version=version,
    description="Fast fractions data type for rational numbers. "
                "Cythonized version of 'fractions.Fraction'.",
    long_description=long_description,
    author="Stefan Behnel",
    author_email="stefan_ml@behnel.de",
    url="https://github.com/scoder/quicktions",
    #bugtrack_url="https://github.com/scoder/quicktions/issues",

    ext_modules=ext_modules,
    packages=['quicktions'],
    package_data={'quicktions':['*.pxd']},
    include_package_data=True,

    classifiers=[
        "Development Status :: 6 - Mature",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: Python Software Foundation License",
    ],
)
