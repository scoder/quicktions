==========
quicktions
==========

Python's ``Fraction`` data type is an excellent way to do exact calculations
with unlimited rational numbers and largely beats ``Decimal`` in terms of
simplicity, accuracy and safety.  Clearly not in terms of speed, though,
given the cdecimal accelerator in Python 3.3+.

``quicktions`` is an adaptation of the original ``fractions`` module
(as included in CPython 3.13) that is compiled and optimised with
`Cython <https://cython.org/>`_ into a fast, native extension module.

Compared to the standard library ``fractions`` module of CPython 3.12,
computations in ``quicktions`` are about 2-4x faster.

Instantiation of a ``Fraction`` in ``quicktions`` is also

- 5-15x faster from a floating point string value (e.g. ``Fraction("123.456789")``)
- 3-5x faster from a floating point value (e.g. ``Fraction(123.456789)``)
- 2-5x faster from an integer numerator-denominator pair (e.g. ``Fraction(123, 456)``)

We provide a set of micro-benchmarks here:

https://github.com/scoder/quicktions/tree/master/benchmark

As of quicktions 1.19, the different number types and implementations compare
as follows in CPython 3.12 (measured on Ubuntu Linux):

.. code-block::

    Average times for all 'create' benchmarks:
    float               :    33.55 us (1.0x)
    Fraction            :   116.02 us (3.5x)
    Decimal             :   132.22 us (3.9x)
    PyFraction          :   361.93 us (10.8x)

    Average times for all 'compute' benchmarks:
    float               :     3.24 us (1.0x)
    Decimal             :    17.28 us (5.3x)
    Fraction            :    77.04 us (23.8x)
    PyFraction          :   166.38 us (51.2x)

While not as fast as the C implemented ``decimal`` module in Python 3,
``quicktions`` is about 15x faster than the Python implemented ``decimal``
module in Python 2.7.

For documentation, see the Python standard library's ``fractions`` module:

https://docs.python.org/3/library/fractions.html
