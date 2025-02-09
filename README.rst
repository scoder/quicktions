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
    float               :    19.69 us (1.0x)
    Fraction            :    58.63 us (3.0x)
    Decimal             :    84.32 us (4.3x)
    PyFraction          :   208.20 us (10.6x)

    Average times for all 'compute' benchmarks:
    float               :     1.79 us (1.0x)
    Decimal             :    10.11 us (5.7x)
    Fraction            :    39.24 us (22.0x)
    PyFraction          :    96.23 us (53.9x)

While not as fast as the C implemented ``decimal`` module in Python 3,
``quicktions`` is about 15x faster than the Python implemented ``decimal``
module in Python 2.7.

For documentation, see the Python standard library's ``fractions`` module:

https://docs.python.org/3/library/fractions.html
