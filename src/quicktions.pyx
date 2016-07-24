## cython: profile=True

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014 Python Software Foundation; All Rights Reserved
#
# Based on the "fractions" module in CPython 3.4/5.
# https://hg.python.org/cpython/file/b18288f24501/Lib/fractions.py
#
# Adapted for efficient Cython compilation by Stefan Behnel.
#

"""
Fast fractions data type for rational numbers.

This is an almost-drop-in replacement for the standard library's
"fractions.Fraction".
"""

from __future__ import division, absolute_import, print_function


__all__ = ['Fraction']

__version__ = '1.3'

cimport cython
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cpython.version cimport PY_MAJOR_VERSION

cdef extern from *:
    ctypedef long Py_hash_t
    cdef long LONG_MAX

cdef object Rational, Decimal, math, numbers, sys, re, operator

from numbers import Rational
from decimal import Decimal
import math
import numbers
import operator
import re
import sys

cdef bint _decimal_supports_integer_ratio = hasattr(Decimal, "as_integer_ratio")  # Py3.6+


cpdef _gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b as a non-negative number.
    """
    # Try doing all computation in C space.  If the numbers are too
    # large at the beginning, retry after each iteration until they
    # are small enough.
    cdef unsigned long long au, bu
    cdef long long ai, bi
    while b:
        try:
            ai, bi = a, b
        except OverflowError:
            pass
        else:
            # switch to C space
            au = abs(ai)
            bu = abs(bi)
            while bu:
                au, bu = bu, au%bu
            # try PyInt downcast in Py2
            if PY_MAJOR_VERSION < 3 and au <= <unsigned long long>LONG_MAX:
                return <long>au
            return au
        a, b = b, a%b
    return abs(a)


# Constants related to the hash implementation;  hash(x) is based
# on the reduction of x modulo the prime _PyHASH_MODULUS.

cdef Py_hash_t _PyHASH_MODULUS
try:
    _PyHASH_MODULUS = sys.hash_info.modulus
except AttributeError:  # pre Py3.2
    # adapted from pyhash.h in Py3.4
    _PyHASH_MODULUS = (<Py_hash_t>1) << (61 if sizeof(Py_hash_t) >= 8 else 31) - 1


# Value to be used for rationals that reduce to infinity modulo
# _PyHASH_MODULUS.
cdef Py_hash_t _PyHASH_INF
try:
    _PyHASH_INF = sys.hash_info.inf
except AttributeError:  # pre Py3.2
    _PyHASH_INF = hash(float('+inf'))


cdef object _parse_rational = re.compile(r"""
    \A\s*                      # optional whitespace at the start, then
    (?P<sign>[-+]?)            # an optional sign, then
    (?=\d|\.\d)                # lookahead for digit or .digit
    (?P<num>\d*)               # numerator (possibly empty)
    (?:                        # followed by
       (?:/(?P<denom>\d+))?    # an optional denominator
    |                          # or
       (?:\.(?P<decimal>\d*))? # an optional fractional part
       (?:E(?P<exp>[-+]?\d+))? # and optional exponent
    )
    \s*\Z                      # and optional whitespace to finish
""", re.VERBOSE | re.IGNORECASE).match


cdef class Fraction:
    """A Rational number.

    Takes a string like '3/2' or '1.5', another Rational instance, a
    numerator/denominator pair, or a float.

    Examples
    --------

    >>> Fraction(10, -8)
    Fraction(-5, 4)
    >>> Fraction(Fraction(1, 7), 5)
    Fraction(1, 35)
    >>> Fraction(Fraction(1, 7), Fraction(2, 3))
    Fraction(3, 14)
    >>> Fraction('314')
    Fraction(314, 1)
    >>> Fraction('-35/4')
    Fraction(-35, 4)
    >>> Fraction('3.1415') # conversion from numeric string
    Fraction(6283, 2000)
    >>> Fraction('-47e-2') # string may include a decimal exponent
    Fraction(-47, 100)
    >>> Fraction(1.47)  # direct construction from float (exact conversion)
    Fraction(6620291452234629, 4503599627370496)
    >>> Fraction(2.25)
    Fraction(9, 4)
    >>> from decimal import Decimal
    >>> Fraction(Decimal('1.47'))
    Fraction(147, 100)

    """
    cdef _numerator
    cdef _denominator
    cdef Py_hash_t _hash

    def __cinit__(self, numerator=0, denominator=None, bint _normalize=True):
        cdef Fraction value
        self._hash = -1
        if denominator is None:
            if type(numerator) is int or type(numerator) is long:
                self._numerator = numerator
                self._denominator = 1
                return

            elif isinstance(numerator, float):
                # Exact conversion
                self._numerator, self._denominator = numerator.as_integer_ratio()
                return

            elif type(numerator) is Fraction:
                self._numerator = (<Fraction>numerator)._numerator
                self._denominator = (<Fraction>numerator)._denominator
                return

            elif isinstance(numerator, basestring):
                # Handle construction from strings.
                m = _parse_rational(numerator)
                if m is None:
                    raise ValueError('Invalid literal for Fraction: %r' %
                                     numerator)
                group = m.group
                numerator = int(group('num') or 0)
                denom = group('denom')
                if denom:
                    denominator = int(denom)
                else:
                    decimal = group('decimal')
                    if decimal:
                        scale = 10 ** <object>len(decimal)
                        numerator = numerator * scale + int(decimal)
                        denominator = scale
                    else:
                        denominator = 1
                    exp = group('exp')
                    if exp:
                        exp = int(exp)
                        if exp >= 0:
                            numerator *= 10**exp
                        else:
                            denominator *= 10**-exp
                if group('sign') == '-':
                    numerator = -numerator
                # fall through to normalisation below

            elif isinstance(numerator, (Fraction, Rational)):
                self._numerator = numerator.numerator
                self._denominator = numerator.denominator
                return

            elif isinstance(numerator, Decimal):
                if _decimal_supports_integer_ratio:
                    # Exact conversion
                    self._numerator, self._denominator = numerator.as_integer_ratio()
                else:
                    value = Fraction.from_decimal(numerator)
                    self._numerator = (<Fraction>value)._numerator
                    self._denominator = (<Fraction>value)._denominator
                return

            else:
                raise TypeError("argument should be a string "
                                "or a Rational instance")

        elif type(numerator) is int is type(denominator):
            pass  # *very* normal case

        elif PY_MAJOR_VERSION < 3 and type(numerator) is long is type(denominator):
            pass  # *very* normal case

        elif type(numerator) is Fraction is type(denominator):
            numerator, denominator = (
                (<Fraction>numerator)._numerator * (<Fraction>denominator)._denominator,
                (<Fraction>denominator)._numerator * (<Fraction>numerator)._denominator
                )

        elif (isinstance(numerator, (Fraction, Rational)) and
                  isinstance(denominator, (Fraction, Rational))):
            numerator, denominator = (
                numerator.numerator * denominator.denominator,
                denominator.numerator * numerator.denominator
                )

        else:
            raise TypeError("both arguments should be "
                            "Rational instances")

        if denominator == 0:
            raise ZeroDivisionError('Fraction(%s, 0)' % numerator)
        if _normalize:
            g = _gcd(numerator, denominator)
            # NOTE: 'is' tests on integers are generally a bad idea, but
            # they are fast and if they fail here, it'll still be correct
            if denominator < 0:
                if g is 1:
                    numerator = -numerator
                    denominator = -denominator
                else:
                    g = -g
            if g is not 1:
                numerator //= g
                denominator //= g
        self._numerator = numerator
        self._denominator = denominator

    @classmethod
    def from_float(cls, f):
        """Converts a finite float to a rational number, exactly.

        Beware that Fraction.from_float(0.3) != Fraction(3, 10).

        """
        try:
            ratio = f.as_integer_ratio()
        except (ValueError, OverflowError, AttributeError):
            pass  # not something we can convert, raise concrete exceptions below
        else:
            return cls(*ratio)

        if isinstance(f, numbers.Integral):
            return cls(f)
        elif not isinstance(f, float):
            raise TypeError("%s.from_float() only takes floats, not %r (%s)" %
                            (cls.__name__, f, type(f).__name__))
        if math.isinf(f):
            raise OverflowError("Cannot convert %r to %s." % (f, cls.__name__))
        raise ValueError("Cannot convert %r to %s." % (f, cls.__name__))

    @classmethod
    def from_decimal(cls, dec):
        """Converts a finite Decimal instance to a rational number, exactly."""
        if isinstance(dec, numbers.Integral):
            dec = Decimal(int(dec))
        elif not isinstance(dec, Decimal):
            raise TypeError(
                "%s.from_decimal() only takes Decimals, not %r (%s)" %
                (cls.__name__, dec, type(dec).__name__))
        if dec.is_infinite():
            raise OverflowError(
                "Cannot convert %s to %s." % (dec, cls.__name__))
        if dec.is_nan():
            raise ValueError("Cannot convert %s to %s." % (dec, cls.__name__))
        sign, digits, exp = dec.as_tuple()
        digits = int(''.join(map(str, digits)))
        if sign:
            digits = -digits
        if exp >= 0:
            return cls(digits * 10 ** exp)
        else:
            return cls(digits, 10 ** -exp)

    def limit_denominator(self, max_denominator=1000000):
        """Closest Fraction to self with denominator at most max_denominator.

        >>> Fraction('3.141592653589793').limit_denominator(10)
        Fraction(22, 7)
        >>> Fraction('3.141592653589793').limit_denominator(100)
        Fraction(311, 99)
        >>> Fraction(4321, 8765).limit_denominator(10000)
        Fraction(4321, 8765)

        """
        # Algorithm notes: For any real number x, define a *best upper
        # approximation* to x to be a rational number p/q such that:
        #
        #   (1) p/q >= x, and
        #   (2) if p/q > r/s >= x then s > q, for any rational r/s.
        #
        # Define *best lower approximation* similarly.  Then it can be
        # proved that a rational number is a best upper or lower
        # approximation to x if, and only if, it is a convergent or
        # semiconvergent of the (unique shortest) continued fraction
        # associated to x.
        #
        # To find a best rational approximation with denominator <= M,
        # we find the best upper and lower approximations with
        # denominator <= M and take whichever of these is closer to x.
        # In the event of a tie, the bound with smaller denominator is
        # chosen.  If both denominators are equal (which can happen
        # only when max_denominator == 1 and self is midway between
        # two integers) the lower bound---i.e., the floor of self, is
        # taken.

        if max_denominator < 1:
            raise ValueError("max_denominator should be at least 1")
        if self._denominator <= max_denominator:
            return Fraction(self)

        p0, q0, p1, q1 = 0, 1, 1, 0
        n, d = self._numerator, self._denominator
        while True:
            a = n//d
            q2 = q0+a*q1
            if q2 > max_denominator:
                break
            p0, q0, p1, q1 = p1, q1, p0+a*p1, q2
            n, d = d, n-a*d

        k = (max_denominator-q0)//q1
        bound1 = Fraction(p0+k*p1, q0+k*q1)
        bound2 = Fraction(p1, q1)
        if abs(bound2 - self) <= abs(bound1-self):
            return bound2
        else:
            return bound1

    property numerator:
        def __get__(a):
            return a._numerator

    property denominator:
        def __get__(a):
            return a._denominator

    def __repr__(self):
        """repr(self)"""
        return '%s(%s, %s)' % (self.__class__.__name__,
                               self._numerator, self._denominator)

    def __str__(self):
        """str(self)"""
        if self._denominator == 1:
            return str(self._numerator)
        else:
            return '%s/%s' % (self._numerator, self._denominator)

    def __add__(a, b):
        """a + b"""
        return _math_op(a, b, _add, 'add')

    def __sub__(a, b):
        """a - b"""
        return _math_op(a, b, _sub, 'sub')

    def __mul__(a, b):
        """a * b"""
        return _math_op(a, b, _mul, 'mul')

    def __div__(a, b):
        """a / b"""
        return _math_op(a, b, _div, 'div')

    def __truediv__(a, b):
        """a / b"""
        return _math_op(a, b, _div, 'truediv')

    def __floordiv__(a, b):
        """a // b"""
        div = a / b
        if PY_MAJOR_VERSION < 3 and isinstance(div, (Fraction, Rational)):
            # trunc(math.floor(div)) doesn't work if the rational is
            # more precise than a float because the intermediate
            # rounding may cross an integer boundary.
            return div.numerator // div.denominator
        else:
            return math.floor(div)

    def __mod__(a, b):
        """a % b"""
        div = a // b
        return a - b * div

    def __divmod__(a, b):
        """divmod(self, other): The pair (self // other, self % other).

        Sometimes this can be computed faster than the pair of
        operations.
        """
        div = a // b
        return (div, a - b * div)

    def __pow__(a, b, x):
        """a ** b

        If b is not an integer, the result will be a float or complex
        since roots are generally irrational. If b is an integer, the
        result will be rational.
        """
        if x is not None:
            return NotImplemented
        if isinstance(a, Fraction):
            # normal call
            if isinstance(b, (int, long, Fraction, Rational)):
                return _pow(a.numerator, a.denominator, b.numerator, b.denominator)
            else:
                return (a.numerator / a.denominator) ** b
        else:
            # reversed call
            bn, bd = b.numerator, b.denominator
            if bd == 1 and bn >= 0:
                # If a is an int, keep it that way if possible.
                return a ** bn

            if isinstance(a, (int, long, Rational)):
                return _pow(a.numerator, a.denominator, bn, bd)

            if bd == 1:
                return a ** bn

            return a ** (bn / bd)

    def __pos__(a):
        """+a: Coerces a subclass instance to Fraction"""
        if type(a) is Fraction:
            return a
        return Fraction(a._numerator, a._denominator, _normalize=False)

    def __neg__(a):
        """-a"""
        return Fraction(-a._numerator, a._denominator, _normalize=False)

    def __abs__(a):
        """abs(a)"""
        return Fraction(abs(a._numerator), a._denominator, _normalize=False)

    def __trunc__(a):
        """trunc(a)"""
        if a._numerator < 0:
            return -(-a._numerator // a._denominator)
        else:
            return a._numerator // a._denominator

    def __floor__(a):
        """math.floor(a)"""
        return a.numerator // a.denominator

    def __ceil__(a):
        """math.ceil(a)"""
        # The negations cleverly convince floordiv to return the ceiling.
        return -(-a.numerator // a.denominator)

    def __round__(self, ndigits=None):
        """round(self, ndigits)

        Rounds half toward even.
        """
        if ndigits is None:
            floor, remainder = divmod(self.numerator, self.denominator)
            if remainder * 2 < self.denominator:
                return floor
            elif remainder * 2 > self.denominator:
                return floor + 1
            # Deal with the half case:
            elif floor % 2 == 0:
                return floor
            else:
                return floor + 1
        shift = 10**abs(ndigits)
        # See _operator_fallbacks.forward to check that the results of
        # these operations will always be Fraction and therefore have
        # round().
        if ndigits > 0:
            return Fraction(round(self * shift), shift)
        else:
            return Fraction(round(self / shift) * shift)

    def __float__(self):
        """float(self) = self.numerator / self.denominator

        It's important that this conversion use the integer's "true"
        division rather than casting one side to float before dividing
        so that ratios of huge integers convert without overflowing.
        """
        return _as_float(self.numerator, self.denominator)

    # Concrete implementations of Complex abstract methods.
    def __complex__(self):
        """complex(self) == complex(float(self), 0)"""
        return complex(float(self))

    # == +self
    real = property(__pos__, doc="Real numbers are their real component.")

    # == 0
    property imag:
        "Real numbers have no imaginary component."
        def __get__(self):
            return 0

    def conjugate(self):
        """Conjugate is a no-op for Reals."""
        return +self

    def __hash__(self):
        """hash(self)"""
        if self._hash != -1:
            return self._hash

        cdef Py_hash_t result

        # Py2 and Py3 use completely different hash functions, we provide both
        if PY_MAJOR_VERSION == 2:
            if self._denominator == 1:
                # Get integers right.
                result = hash(self._numerator)
            # Expensive check, but definitely correct.
            if self == float(self):
                result = hash(float(self))
            else:
                # Use tuple's hash to avoid a high collision rate on
                # simple fractions.
                result = hash((self._numerator, self._denominator))
            self._hash = result
            return result

        # In order to make sure that the hash of a Fraction agrees
        # with the hash of a numerically equal integer, float or
        # Decimal instance, we follow the rules for numeric hashes
        # outlined in the documentation.  (See library docs, 'Built-in
        # Types').

        # dinv is the inverse of self._denominator modulo the prime
        # _PyHASH_MODULUS, or 0 if self._denominator is divisible by
        # _PyHASH_MODULUS.
        dinv = pow(self._denominator, _PyHASH_MODULUS - 2, _PyHASH_MODULUS)
        if not dinv:
            result = _PyHASH_INF
        else:
            result = abs(self._numerator) * dinv % _PyHASH_MODULUS
        if self._numerator < 0:
            result = -result
            if result == -1:
                result = -2
        self._hash = result
        return result

    def __richcmp__(a, b, int op):
        if isinstance(a, Fraction):
            if op == Py_EQ:
                return (<Fraction>a)._eq(b)
            elif op == Py_NE:
                result = (<Fraction>a)._eq(b)
                return NotImplemented if result is NotImplemented else not result
            elif op == Py_LT:
                pyop = operator.lt
            elif op == Py_GT:
                pyop = operator.gt
            elif op == Py_LE:
                pyop = operator.le
            elif op == Py_GE:
                pyop = operator.ge
            else:
                return NotImplemented
        else:
            a, b = b, a
            if op == Py_EQ:
                return (<Fraction>a)._eq(b)
            elif op == Py_NE:
                result = (<Fraction>a)._eq(b)
                return NotImplemented if result is NotImplemented else not result
            elif op == Py_LT:
                pyop = operator.ge
            elif op == Py_GT:
                pyop = operator.le
            elif op == Py_LE:
                pyop = operator.gt
            elif op == Py_GE:
                pyop = operator.lt
            else:
                return NotImplemented
        return (<Fraction>a)._richcmp(b, pyop)

    @cython.final
    cdef _eq(a, b):
        if type(b) is int or type(b) is long:
            return a._numerator == b and a._denominator == 1
        if type(b) is Fraction:
            return (a._numerator == (<Fraction>b)._numerator and
                    a._denominator == (<Fraction>b)._denominator)
        if isinstance(b, Rational):
            return (a._numerator == b.numerator and
                    a._denominator == b.denominator)
        if isinstance(b, numbers.Complex) and b.imag == 0:
            b = b.real
        if isinstance(b, float):
            if math.isnan(b) or math.isinf(b):
                # comparisons with an infinity or nan should behave in
                # the same way for any finite a, so treat a as zero.
                return 0.0 == b
            else:
                return a == a.from_float(b)
        return NotImplemented

    @cython.final
    cdef _richcmp(self, other, op):
        """Helper for comparison operators, for internal use only.

        Implement comparison between a Rational instance `self`, and
        either another Rational instance or a float `other`.  If
        `other` is not a Rational instance or a float, return
        NotImplemented. `op` should be one of the six standard
        comparison operators.

        """
        # convert other to a Rational instance where reasonable.
        if isinstance(other, (int, long)):
            return op(self._numerator, self._denominator * other)
        if isinstance(other, (Fraction, Rational)):
            return op(self._numerator * other.denominator,
                      self._denominator * other.numerator)
        if isinstance(other, float):
            if math.isnan(other) or math.isinf(other):
                return op(0.0, other)
            else:
                return op(self, self.from_float(other))
        # comparisons with complex should raise a TypeError, for consistency
        # with int<->complex, float<->complex, and complex<->complex comparisons.
        if PY_MAJOR_VERSION < 3 and isinstance(other, complex):
            raise TypeError("no ordering relation is defined for complex numbers")
        return NotImplemented

    def __bool__(self):
        """a != 0"""
        return self._numerator != 0

    # support for pickling, copy, and deepcopy

    def __reduce__(self):
        return (type(self), (str(self),))

    def __copy__(self):
        if type(self) is Fraction:
            return self     # I'm immutable; therefore I am my own clone
        return type(self)(self._numerator, self._denominator)

    def __deepcopy__(self, memo):
        if type(self) is Fraction:
            return self     # My components are also immutable
        return type(self)(self._numerator, self._denominator)


cdef _pow(an, ad, bn, bd):
    if bd == 1:
        if bn >= 0:
            return Fraction(an ** bn,
                            ad ** bn,
                            _normalize=False)
        else:
            return Fraction(ad ** -bn,
                            an ** -bn,
                            _normalize=False)
    else:
        # A fractional power will generally produce an
        # irrational number.
        if PY_MAJOR_VERSION >= 3:
            return (an / ad) ** (bn / bd)
        else:
            return float(an / ad) ** (bn / bd)


cdef _as_float(numerator, denominator):
    return numerator / denominator


"""
In general, we want to implement the arithmetic operations so
that mixed-mode operations either call an implementation whose
author knew about the types of both arguments, or convert both
to the nearest built in type and do the operation there. In
Fraction, that means that we define __add__ and __radd__ as:

    def __add__(self, other):
        # Both types have numerators/denominator attributes,
        # so do the operation directly
        if isinstance(other, (int, Fraction)):
            return Fraction(self.numerator * other.denominator +
                            other.numerator * self.denominator,
                            self.denominator * other.denominator)
        # float and complex don't have those operations, but we
        # know about those types, so special case them.
        elif isinstance(other, float):
            return float(self) + other
        elif isinstance(other, complex):
            return complex(self) + other
        # Let the other type take over.
        return NotImplemented

    def __radd__(self, other):
        # radd handles more types than add because there's
        # nothing left to fall back to.
        if isinstance(other, Rational):
            return Fraction(self.numerator * other.denominator +
                            other.numerator * self.denominator,
                            self.denominator * other.denominator)
        elif isinstance(other, Real):
            return float(other) + float(self)
        elif isinstance(other, Complex):
            return complex(other) + complex(self)
        return NotImplemented


There are 5 different cases for a mixed-type addition on
Fraction. I'll refer to all of the above code that doesn't
refer to Fraction, float, or complex as "boilerplate". 'r'
will be an instance of Fraction, which is a subtype of
Rational (r : Fraction <: Rational), and b : B <:
Complex. The first three involve 'r + b':

    1. If B <: Fraction, int, float, or complex, we handle
       that specially, and all is well.
    2. If Fraction falls back to the boilerplate code, and it
       were to return a value from __add__, we'd miss the
       possibility that B defines a more intelligent __radd__,
       so the boilerplate should return NotImplemented from
       __add__. In particular, we don't handle Rational
       here, even though we could get an exact answer, in case
       the other type wants to do something special.
    3. If B <: Fraction, Python tries B.__radd__ before
       Fraction.__add__. This is ok, because it was
       implemented with knowledge of Fraction, so it can
       handle those instances before delegating to Real or
       Complex.

The next two situations describe 'b + r'. We assume that b
didn't know about Fraction in its implementation, and that it
uses similar boilerplate code:

    4. If B <: Rational, then __radd_ converts both to the
       builtin rational type (hey look, that's us) and
       proceeds.
    5. Otherwise, __radd__ tries to find the nearest common
       base ABC, and fall back to its builtin type. Since this
       class doesn't subclass a concrete type, there's no
       implementation to fall back to, so we need to try as
       hard as possible to return an actual value, or the user
       will get a TypeError.
"""


cdef _add(an, ad, bn, bd):
    """a + b"""
    return Fraction(an * bd + bn * ad, ad * bd)

cdef _sub(an, ad, bn, bd):
    """a - b"""
    return Fraction(an * bd - bn * ad, ad * bd)

cdef _mul(an, ad, bn, bd):
    """a * b"""
    return Fraction(an * bn, ad * bd)

cdef _div(an, ad, bn, bd):
    """a / b"""
    return Fraction(an * bd, ad * bn)


ctypedef object (*math_func)(an, ad, bn, bd)


cdef _math_op(a, b, math_func monomorphic_operator, str pyoperator_name):
    if isinstance(a, Fraction):
        return forward(a, b, monomorphic_operator, pyoperator_name)
    else:
        return reverse(a, b, monomorphic_operator, pyoperator_name)


cdef forward(a, b, math_func monomorphic_operator, str pyoperator_name):
    an, ad = (<Fraction>a)._numerator, (<Fraction>a)._denominator
    if type(b) is Fraction:
        return monomorphic_operator(an, ad, (<Fraction>b)._numerator, (<Fraction>b)._denominator)
    elif isinstance(b, (int, long)):
        return monomorphic_operator(an, ad, b, 1)
    elif isinstance(b, (Fraction, Rational)):
        return monomorphic_operator(an, ad, b.numerator, b.denominator)
    elif isinstance(b, float):
        return getattr(operator, pyoperator_name)(_as_float(an, ad), b)
    elif isinstance(b, complex):
        return getattr(operator, pyoperator_name)(complex(a), b)
    else:
        return NotImplemented


cdef reverse(a, b, math_func monomorphic_operator, str pyoperator_name):
    bn, bd = (<Fraction>b)._numerator, (<Fraction>b)._denominator
    if isinstance(a, (int, long)):
        return monomorphic_operator(a, 1, bn, bd)
    elif isinstance(a, Rational):
        return monomorphic_operator(a.numerator, a.denominator, bn, bd)
    elif isinstance(a, numbers.Real):
        return getattr(operator, pyoperator_name)(float(a), _as_float(bn, bd))
    elif isinstance(a, numbers.Complex):
        return getattr(operator, pyoperator_name)(complex(a), complex(b))
    else:
        return NotImplemented


Rational.register(Fraction)
