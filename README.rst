==========
quicktions
==========

Python's ``Fraction`` data type is an excellent way to do exact calculations
with unlimited rational numbers and largely beats ``Decimal`` in terms of
simplicity, accuracy and safety.  Clearly not in terms of speed, though,
given the cdecimal accelerator in Python 3.3+.

``quicktions`` is an adaptation of the original ``fractions`` module
(as included in CPython 3.13a3) that is compiled and optimised with
`Cython <https://cython.org/>`_ into a fast, native extension module.

Compared to the standard library ``fractions`` module of CPython,
computations in ``quicktions`` are about

- 10x faster in Python 2.7 and 3.4
- 6x faster in Python 3.5
- 3-4x faster in Python 3.10

Compared to the ``fractions`` module in CPython 3.10, instantiation of a
``Fraction`` in ``quicktions`` is also

- 5-15x faster from a floating point string value (e.g. ``Fraction("123.456789")``)
- 3-5x faster from a floating point value (e.g. ``Fraction(123.456789)``)
- 2-4x faster from an integer numerator-denominator pair (e.g. ``Fraction(123, 456)``)

We provide a set of micro-benchmarks here:

https://github.com/scoder/quicktions/tree/master/benchmark

As of quicktions 1.12, the different number types and implementations compare
as follows in CPython 3.10:

.. code-block::

    Average times for all 'create' benchmarks:
    float               :    36.17 us (1.0x)
    Decimal             :   111.71 us (3.1x)
    Fraction            :   111.98 us (3.1x)
    PyFraction          :   398.80 us (11.0x)

    Average times for all 'compute' benchmarks:
    float               :     4.53 us (1.0x)
    Decimal             :    16.62 us (3.7x)
    Fraction            :    72.91 us (16.1x)
    PyFraction          :   251.93 us (55.6x)

While not as fast as the C implemented ``decimal`` module in Python 3,
``quicktions`` is about 15x faster than the Python implemented ``decimal``
module in Python 2.7.

For documentation, see the Python standard library's ``fractions`` module:

https://docs.python.org/3/library/fractions.html
