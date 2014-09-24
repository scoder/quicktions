
import sys
import re
from distutils.core import setup, Extension

ext_modules = [
    Extension("quicktions", ["src/quicktions.pyx"]),
]

try:
    sys.argv.remove("--with-cython")
except ValueError:
    cythonize = None
else:
    try:
        from Cython.Build import cythonize
        import Cython.Compiler.Options as cython_options
        cython_options.annotate = True
    except ImportError:
        cythonize = None
    else:
        ext_modules = cythonize(ext_modules)

if cythonize is None:
    for ext_module in ext_modules:
        ext_module.sources[:] = [m.replace('.pyx', '.c') for m in ext_module.sources]



with open('src/quicktions.pyx') as f:
    version = re.search("__version__\s*=\s*'([^']+)'", f.read(2048)).group(1)

setup(
    name="quicktions",
    version=version,
    description="Fast fractions data type for rational numbers. "
                "Cythonized version of 'fractions.Fraction'.",
    long_description="""\
    Python's Fraction data type is an excellent way to do exact money
    calculations and largely beats Decimal in terms of simplicity,
    accuracy and safety.  Clearly not in terms of speed, though.

    This is an adaptation of the original module (as included in
    CPython 3.4) that is compiled and optimised with Cython into a
    fast, native extension module.

    Compared to the standard library 'fractions' module in Py2.7 and
    Py3.4, 'quicktions' is about 10x faster.
    """,
    author="Stefan Behnel",
    author_email="stefan_ml@behnel.de",

    ext_modules=ext_modules,
    package_dir={'': 'src'},

    classifiers=[
        "Development Status :: 4 - Beta",
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
