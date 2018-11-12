ChangeLog
=========

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
