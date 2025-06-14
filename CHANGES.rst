ChangeLog
=========

1.22 (2025-??-??)
-----------------

* A choice of different GCD implementations is available via ``quicktions.use_gcd_impl()``.
  The fastest one on the current machine is chosen at import time.


1.21 (2025-06-13)
-----------------

* A serious parser bug could accidentally concatenate numerator and denominator
  as final denominator when parsing "x/y" where x or y are close to ``sys.maxsize``,
  thus returning a ``Fraction("x/xy")``.

* MSVC and clang now also benefit from fast "count trailing zeroes" intrinsics.


1.20 (2025-06-13)
-----------------

* ``quicktions`` is compatible with freethreading Python (3.13+).

* Accept leading zeros in precision/width for Fraction's formatting, following
  https://github.com/python/cpython/pull/130663

* In line with Python's ``Fraction``, quicktions now raises a ``ValueError``
  (instead of an ``OverflowError``) when exceeding parser limits, following
  https://github.com/python/cpython/pull/134010

* Call ``__rpow__`` in ternary ``pow()`` if necessary, following
  https://github.com/python/cpython/pull/130251

* Built using Cython 3.1.2.


1.19 (2024-11-29)
-----------------

* Support for Python 2.7 as well as 3.7 and earlier has been removed.

* Generally use ``.as_integer_ratio()`` in the constructor if available.
  https://github.com/python/cpython/pull/120271

* Add a classmethod ``.from_number()`` that requires a number argument, not a string.
  https://github.com/python/cpython/pull/121800

* Mixed calculations with other ``Rational`` classes could return the wrong type.
  https://github.com/python/cpython/issues/119189

* In mixed calculations with ``complex``, the Fraction is now converted to ``float``
  instead of ``complex`` to avoid certain corner cases in complex calculation.
  https://github.com/python/cpython/pull/119839

* Using ``complex`` numbers in division shows better tracebacks.
  https://github.com/python/cpython/pull/102842

* Subclass instantiations and calculations could fail in some cases.


1.18 (2024-04-03)
-----------------

* New binary wheels were added built with gcc 12 (manylinux_2_28).

* x86_64 wheels now require SSE4.2.

* Built using Cython 3.0.10.


1.17 (2024-03-24)
-----------------

* Math operations were sped up by inlined binary GCD calculation.


1.16 (2024-01-10)
-----------------

* Formatting support was improved, following CPython 3.13a3 as of
  https://github.com/python/cpython/pull/111320

* Add support for Python 3.13 by using Cython 3.0.8 and calling ``math.gcd()``.


1.15 (2023-08-27)
-----------------

* Add support for Python 3.12 by using Cython 3.0.2.


1.14 (2023-03-19)
-----------------

* Implement ``__format__`` for ``Fraction``, following
  https://github.com/python/cpython/pull/100161

* Implement ``Fraction.is_integer()``, following
  https://github.com/python/cpython/issues/100488

* ``Fraction.limit_denominator()`` is faster, following
  https://github.com/python/cpython/pull/93730

* Internal creation of result Fractions is about 10% faster if the calculated
  numerator/denominator pair is already normalised, following
  https://github.com/python/cpython/pull/101780

* Built using Cython 3.0.0b1.


1.13 (2022-01-11)
-----------------

* Parsing very long numbers from a fraction string was very slow, even slower
  than ``fractions.Fraction``.  The parser is now faster in all cases (and
  still much faster for shorter numbers).

* ``Fraction`` did not implement ``__int__``.
  https://bugs.python.org/issue44547


1.12 (2022-01-07)
-----------------

* Faster and more space friendly pickling and unpickling.
  https://bugs.python.org/issue44154

* Algorithmically faster arithmetic for large denominators, although slower for
  small fraction components.
  https://bugs.python.org/issue43420
  Original patch for CPython by Sergey B. Kirpichev and Raymond Hettinger.

* Make sure ``bool(Fraction)`` always returns a ``bool``.
  https://bugs.python.org/issue39274

* Built using Cython 3.0.0a10.


1.11 (2019-12-19)
-----------------

* Fix ``OverflowError`` when parsing string values with long decimal parts.


1.10 (2019-08-23)
-----------------

* ``hash(fraction)`` is substantially faster in Py3.8+, following an optimisation
  in CPython 3.9 (https://bugs.python.org/issue37863).

* New method ``fraction.as_integer_ratio()``.


1.9 (2018-12-26)
----------------

* Substantially faster normalisation (and therefore instantiation) in Py3.5+.

* ``//`` (floordiv) now follows the expected rounding behaviour when used with
  floats (by converting to float first), and is much faster for integer operations.

* Fix return type of divmod(), where the first item should be an integer.

* Further speed up mod and divmod operations.


1.8 (2018-12-26)
----------------

* Faster mod and divmod calculation.


1.7 (2018-10-16)
----------------

* Faster normalisation and fraction string parsing.

* Add support for Python 3.7.

* Built using Cython 0.29.


1.6 (2018-03-23)
----------------

* Speed up Fraction creation from a string value by 3-5x.

* Built using Cython 0.28.1.


1.5 (2017-10-22)
----------------

* Result of power operator (``**``) was not normalised for negative values.

* Built using Cython 0.27.2.


1.4 (2017-09-16)
----------------

* Rebuilt using Cython 0.26.1 to improve support of Python 3.7.


1.3 (2016-07-24)
----------------

* repair the faster instantiation from Decimal values in Python 3.6

* avoid potential glitch for certain large numbers in normalisation under Python 2.x


1.2 (2016-04-08)
----------------

* change hash function in Python 2.x to match that of ``fractions.Fraction``


1.1 (2016-03-29)
----------------

* faster instantiation from float values

* faster instantiation from Decimal values in Python 3.6


1.0 (2015-09-10)
----------------

* ``Fraction.imag`` property could return non-zero

* parsing strings with long fraction parts could use an incorrect scale


0.7 (2014-10-09)
----------------

* faster instantiation from float and string values

* fix test in Python 2.x


0.6 (2014-10-09)
----------------

* faster normalisation (and thus instantiation)


0.5 (2014-10-06)
----------------

* faster math operations


0.4 (2014-10-06)
----------------

* enable legacy division support in Python 2.x


0.3 (2014-10-05)
----------------

* minor behavioural fixes in corner cases under Python 2.x
  (now passes all test in Py2.7 as well)


0.2 (2014-10-03)
----------------

* cache hash value of Fractions


0.1 (2014-09-24)
----------------

* initial public release
