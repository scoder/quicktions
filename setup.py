
import os
import re
import sys

from setuptools import setup, Extension


try:
    sys.argv.remove("--with-profile")
except ValueError:
    enable_profiling = False
else:
    enable_profiling = True

enable_coverage = os.environ.get("WITH_COVERAGE") == "1"
force_rebuild = os.environ.get("FORCE_REBUILD") == "1"

def check_limited_api_option(value):
    if not value:
        return None
    value = value.lower()
    if value == "true":
        # The default Limited API version is 3.9, unless we're on a lower Python version
        # (which is mainly for the sake of testing 3.8 on the CI)
        if sys.version_info >= (3, 9):
            return (3, 9)
        else:
            return sys.version_info[:2]
    if value == 'false':
        return None
    major, minor = value.split('.', 1)
    return (int(major), int(minor))

extra_setup_args = {}
c_defines = []

option_limited_api = check_limited_api_option(os.environ.get("QUICKTIONS_LIMITED_API"))
if option_limited_api:
    c_defines.append(('Py_LIMITED_API', f'0x{option_limited_api[0]:02x}{option_limited_api[1]:02x}0000'))

    setup_options = extra_setup_args.setdefault('options', {})
    bdist_wheel_options = setup_options.setdefault('bdist_wheel', {})
    bdist_wheel_options['py_limited_api'] = f'cp{option_limited_api[0]}{option_limited_api[1]}'


ext_modules = [
    Extension(
        "quicktions",
        ["src/quicktions.pyx"],
        py_limited_api=True if option_limited_api else False,
        define_macros=c_defines,
    ),
]


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
    license="PSF-2.0",

    ext_modules=ext_modules,
    package_dir={'': 'src'},

    classifiers=[
        "Development Status :: 6 - Mature",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
    ],
    **extra_setup_args,
)
