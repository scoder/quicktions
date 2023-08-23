
import os
import re
import sys

try:
    import setuptools
except ImportError:
    pass

from distutils.core import setup, Extension


ext_modules = [
    Extension("quicktions", ["src/quicktions.pyx"]),
]

try:
    sys.argv.remove("--with-profile")
except ValueError:
    enable_profiling = False
else:
    enable_profiling = True

enable_coverage = os.environ.get("WITH_COVERAGE") == "1"
force_rebuild = os.environ.get("FORCE_REBUILD") == "1"

try:
    sys.argv.remove("--with-cython")
except ValueError:
    pass  # legacy option

cythonize = None
try:
    sys.argv.remove("--no-cython")
except ValueError:
    try:
        from Cython.Build import cythonize
        from Cython import __version__ as cython_version
        import Cython.Compiler.Options as cython_options
        cython_options.annotate = True
    except ImportError:
        print("Cython not found, building without Cython")
        cythonize = None
    else:
        print("Building with Cython %s" % cython_version)
        compiler_directives = {}
        if enable_profiling:
            compiler_directives['profile'] = True
        if enable_coverage:
            compiler_directives['linetrace'] = True
        ext_modules = cythonize(
            ext_modules, compiler_directives=compiler_directives, force=force_rebuild)

if cythonize is None:
    for ext_module in ext_modules:
        ext_module.sources[:] = [m.replace('.pyx', '.c') for m in ext_module.sources]
elif enable_coverage:
    for ext_module in ext_modules:
        ext_module.extra_compile_args += [
            "-DCYTHON_TRACE_NOGIL=1",
        ]

if sys.platform == "darwin":
    try:
        if int(os.environ.get("MACOSX_DEPLOYMENT_TARGET", "0").split(".", 1)[0]) >= 11:
            if "-arch" not in os.environ.get("CFLAGS", ""):
                os.environ["CFLAGS"] = os.environ.get("CFLAGS", "") + " -arch arm64 -arch x86_64"
                os.environ["LDFLAGS"] = os.environ.get("LDFLAGS", "") + " -arch arm64 -arch x86_64"
    except ValueError:
        pass  # probably cannot parse "MACOSX_DEPLOYMENT_TARGET"


with open('src/quicktions.pyx') as f:
    version = re.search(r"__version__\s*=\s*'([^']+)'", f.read(2048)).group(1)

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
    package_dir={'': 'src'},

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
