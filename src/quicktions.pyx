# cython: language_level=3
## cython: profile=True
# cython: freethreading_compatible=True

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014 Python Software Foundation; All Rights Reserved
#
# Based on the "fractions" module in CPython 3.4+.
# https://hg.python.org/cpython/file/b18288f24501/Lib/fractions.py
#
# Updated to match the recent development in CPython.
# https://github.com/python/cpython/blob/main/Lib/fractions.py
#
# Adapted for efficient Cython compilation by Stefan Behnel.
#

"""
Fast fractions data type for rational numbers.

This is an almost-drop-in replacement for the standard library's
"fractions.Fraction".
"""

__all__ = ['Fraction']

__version__ = '1.21'

cimport cython
from cpython.unicode cimport Py_UNICODE_TODECIMAL
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cpython.long cimport PyLong_FromString

cdef extern from *:
    cdef long LONG_MAX, INT_MAX
    cdef long long PY_LLONG_MIN, PY_LLONG_MAX
    cdef long long MAX_SMALL_NUMBER "(PY_LLONG_MAX / 100)"

cdef object Rational, Integral, Real, Complex, Decimal, math, operator, re
cdef object PY_MAX_ULONGLONG = (<unsigned long long> PY_LLONG_MAX) * 2 + 1

from numbers import Rational, Integral, Real, Complex
from decimal import Decimal
import math
import operator
import re

cdef object _operator_index = operator.index
cdef object math_gcd = math.gcd

# Cache widely used 10**x int objects.
# Py3.12/Ubuntu64: sys.getsizeof(tuple[58]) == 512 bytes, tuple[91] == 768, tuple[123] == 1024
DEF CACHED_POW10 = 91

cdef tuple _cache_pow10():
    cdef int i
    in_ull = True
    l = []
    x = 1
    for i in range(CACHED_POW10):
        l.append(x)
        if in_ull:
            try:
                _C_POW_10[i] = x
            except OverflowError:
                in_ull = False
        x *= 10
    return tuple(l)

cdef unsigned long long[CACHED_POW10] _C_POW_10
cdef tuple POW_10 = _cache_pow10()


cdef unsigned long long _c_pow10(Py_ssize_t i):
    return _C_POW_10[i]


cdef pow10(long long i):
    if 0 <= i < CACHED_POW10:
        return POW_10[i]
    else:
        return 10 ** (<object> i)


# Half-private GCD implementation.

cdef extern from *:
    """
    #if defined(__Pyx_PyLong_DigitCount)
      #define __Quicktions_HAS_ISLONGLONG  (1)
      #define __Quicktions_PyLong_IsLongLong(x)  (__Pyx_PyLong_DigitCount(x) <= 2)
    #else
      #define __Quicktions_HAS_ISLONGLONG  (0)
      #define __Quicktions_PyLong_IsLongLong(x)  (0)
    #endif

    #if PY_VERSION_HEX >= 0x030c00a5 && defined(PyUnstable_Long_IsCompact) && defined(PyUnstable_Long_CompactValue)
        #define __Quicktions_PyLong_IsCompact(x)  PyUnstable_Long_IsCompact((PyLongObject*) (x))
      #if CYTHON_COMPILING_IN_CPYTHON
        #define __Quicktions_PyLong_CompactValueUnsigned(x)  ((unsigned long long) (((PyLongObject*)x)->long_value.ob_digit[0]))
      #else
        #define __Quicktions_PyLong_CompactValueUnsigned(x)  ((unsigned long long) PyUnstable_Long_CompactValue((PyLongObject*) (x))))
      #endif
    #elif PY_VERSION_HEX < 0x030c0000 && CYTHON_COMPILING_IN_CPYTHON
        #define __Quicktions_PyLong_IsCompact(x)     (Py_SIZE(x) == 0 || Py_SIZE(x) == 1 || Py_SIZE(x) == -1)
        #define __Quicktions_PyLong_CompactValueUnsigned(x)  ((unsigned long long) ((Py_SIZE(x) == 0) ? 0 : (((PyLongObject*)x)->ob_digit)[0]))
    #else
        #define __Quicktions_PyLong_IsCompact(x)     (0)
        #define __Quicktions_PyLong_CompactValueUnsigned(x)  (0U)
    #endif
    #if PY_VERSION_HEX >= 0x030d0000 || !CYTHON_COMPILING_IN_CPYTHON
        #define _PyLong_GCD(a, b) (NULL)
    #endif

    #ifdef __has_builtin
      #if __has_builtin(__builtin_ctzg)
        #define __Quicktions_HAS_FAST_CTZ_uint  (1)
        #define __Quicktions_trailing_zeros_uint(x)    __builtin_ctzg(x)
        #define __Quicktions_HAS_FAST_CTZ_ulong  (1)
        #define __Quicktions_trailing_zeros_ulong(x)   __builtin_ctzg(x)
        #define __Quicktions_HAS_FAST_CTZ_ullong  (1)
        #define __Quicktions_trailing_zeros_ullong(x)  __builtin_ctzg(x)
      #else
        #if __has_builtin(__builtin_ctz)
            #define __Quicktions_HAS_FAST_CTZ_uint  (1)
            #define __Quicktions_trailing_zeros_uint(x)    __builtin_ctz(x)
        #endif
        #if __has_builtin(__builtin_ctzl)
            #define __Quicktions_HAS_FAST_CTZ_ulong  (1)
            #define __Quicktions_trailing_zeros_ulong(x)   __builtin_ctzl(x)
        #endif
        #if __has_builtin(__builtin_ctzll)
            #define __Quicktions_HAS_FAST_CTZ_ullong  (1)
            #define __Quicktions_trailing_zeros_ullong(x)  __builtin_ctzll(x)
        #endif
      #endif
    #elif defined(_MSC_VER) && SIZEOF_INT == 4
        /* Typical Windows64 config (Win32 does not define "_BitScanForward64"). */
        #define __Quicktions_HAS_FAST_CTZ_uint  (1)
        #pragma intrinsic(_BitScanForward)
        static CYTHON_INLINE int __Quicktions_trailing_zeros_uint(uint32_t x) {
            unsigned long bits;
            _BitScanForward(&bits, x);
            return (int) bits;
        }
        #if SIZEOF_LONG == 4
          #define __Quicktions_HAS_FAST_CTZ_ulong  (1)
          #define __Quicktions_trailing_zeros_ulong(x)  __Quicktions_trailing_zeros_uint(x)
        #endif

        /* Win32 does not define "_BitScanForward64". */
        #if defined(_WIN64) && SIZEOF_LONG_LONG == 8
          #define __Quicktions_HAS_FAST_CTZ_ullong  (1)
          #pragma intrinsic(_BitScanForward64)
          static CYTHON_INLINE int __Quicktions_trailing_zeros_ullong(uint64_t x) {
              unsigned long bits;
              _BitScanForward64(&bits, x);
              return (int) bits;
          }
          #if SIZEOF_LONG == 8
            #define __Quicktions_HAS_FAST_CTZ_ulong  (1)
            #define __Quicktions_trailing_zeros_ulong(x)  __Quicktions_trailing_zeros_ullong(x)
          #endif
        #endif
    #endif

    #if !defined(__Quicktions_HAS_FAST_CTZ_uint)
        #define __Quicktions_HAS_FAST_CTZ_uint  (0)
        #define __Quicktions_trailing_zeros_uint(x)    (0)
    #endif
    #if !defined(__Quicktions_HAS_FAST_CTZ_ulong)
        #define __Quicktions_HAS_FAST_CTZ_ulong  (0)
        #define __Quicktions_trailing_zeros_ulong(x)   (0)
    #endif
    #if !defined(__Quicktions_HAS_FAST_CTZ_ullong)
        #define __Quicktions_HAS_FAST_CTZ_ullong  (0)
        #define __Quicktions_trailing_zeros_ullong(x)  (0)
    #endif
    """
    bint PyLong_IsCompact "__Quicktions_PyLong_IsCompact" (x)
    unsigned long long PyLong_CompactValueUnsigned "__Quicktions_PyLong_CompactValueUnsigned" (x)
    const unsigned long long PY_ULLONG_MAX

    bint PyLong_IsLongLong "__Quicktions_PyLong_IsLongLong" (x)
    const bint HAS_ISLONGLONG "__Quicktions_HAS_ISLONGLONG"

    # CPython 3.5-3.12 has a fast PyLong GCD implementation that we can use.
    # In CPython 3.13, math.gcd() is fast enough to call it directly.
    const bint HAS_FAST_MATH_GCD  "(PY_VERSION_HEX >= 0x030d0000)"
    const bint HAS_OLD_PYLONG_GCD "(CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX < 0x030d0000)"
    object _PyLong_GCD(object a, object b)

    const bint HAS_FAST_CTZ_uint   "__Quicktions_HAS_FAST_CTZ_uint"
    int trailing_zeros_uint "__Quicktions_trailing_zeros_uint" (unsigned int x)
    const bint HAS_FAST_CTZ_ulong  "__Quicktions_HAS_FAST_CTZ_ulong"
    int trailing_zeros_ulong "__Quicktions_trailing_zeros_ulong" (unsigned long x)
    const bint HAS_FAST_CTZ_ullong "__Quicktions_HAS_FAST_CTZ_ullong"
    int trailing_zeros_ullong "__Quicktions_trailing_zeros_ullong" (unsigned long long x)


def _gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b as a non-negative number.
    """
    if not isinstance(a, int):
        raise ValueError(f"Expected int, got {type(a).__name__}")
    if not isinstance(b, int):
        raise ValueError(f"Expected int, got {type(b).__name__}")

    return _igcd(int(a), int(b))


cdef _igcd(a, b):
    if HAS_ISLONGLONG:
        if PyLong_IsLongLong(a) and PyLong_IsLongLong(b):
            return _c_gcd(<unsigned long long> a, <unsigned long long> b)
    elif PyLong_IsCompact(a) and PyLong_IsCompact(b):
        return _c_gcd(PyLong_CompactValueUnsigned(a), PyLong_CompactValueUnsigned(b))

    if HAS_FAST_MATH_GCD:
        return math_gcd(a, b)
    if HAS_OLD_PYLONG_GCD:
        return _PyLong_GCD(a, b)

    return _gcd_fallback(int(a), int(b))


ctypedef unsigned long long ullong
ctypedef unsigned long ulong
ctypedef unsigned int uint

ctypedef fused cunumber:
    ullong
    ulong
    uint


cdef ullong _abs(long long x) noexcept:
    if x == PY_LLONG_MIN:
        return (<ullong>PY_LLONG_MAX) + 1
    return abs(x)


@cython.cdivision(True)
cdef cunumber _euclid_gcd(cunumber a, cunumber b) noexcept:
    """Euclid's GCD algorithm"""
    while b:
        a, b = b, a%b
    return a


cdef inline int trailing_zeros(cunumber x) noexcept:
    if cunumber is uint:
        return trailing_zeros_uint(x)
    elif cunumber is ulong:
        return trailing_zeros_ulong(x)
    else:
        return trailing_zeros_ullong(x)


cdef cunumber _binary_gcd(cunumber a, cunumber b) noexcept:
    # See https://en.wikipedia.org/wiki/Binary_GCD_algorithm
    if not a:
        return b
    if not b:
        return a

    cdef int az = trailing_zeros(a)
    a >>= az
    cdef int bz = trailing_zeros(b)
    b >>= bz

    cdef int shift = min(az, bz)

    while True:
        if a > b:
            a, b = b, a
        b -= a
        if not b:
            return a << shift
        b >>= trailing_zeros(b)


@cython.cdivision(True)
cdef cunumber _hybrid_binary_gcd(cunumber a, cunumber b) noexcept:
    # See https://lemire.me/blog/2024/04/13/greatest-common-divisor-the-extended-euclidean-algorithm-and-speed/
    if a < b:
        a,b = b,a
    if not b:
        return a
    a %= b
    if not a:
        return b

    cdef int az = trailing_zeros(a)
    cdef int bz = trailing_zeros(b)
    cdef int shift = min(az, bz)
    a >>= az
    b >>= bz

    if cunumber is uint:
        diff: cython.int
    elif cunumber is ulong:
        diff: cython.long
    else:
        diff: cython.longlong

    while True:
        diff = a - b
        if a > b:
            a,b = b,diff
        else:
            b -= a
        if not b:
            return a << shift
        b >>= trailing_zeros(diff)


@cython.cdivision(True)
cdef cunumber _hybrid_binary_gcd2(cunumber a, cunumber b) noexcept:
    # See https://en.algorithmica.org/hpc/algorithms/gcd/
    # See https://lemire.me/blog/2024/04/13/greatest-common-divisor-the-extended-euclidean-algorithm-and-speed/
    if a < b:
        a,b = b,a
    if not b:
        return a
    a %= b
    if not a:
        return b

    cdef int az = trailing_zeros(a)
    cdef int bz = trailing_zeros(b)
    cdef int shift = min(az, bz)
    b >>= bz

    cdef cunumber diff

    while a != 0:
        a >>= az
        diff = b - a if b > a else a - b
        if b > a:
            b = a
        az = trailing_zeros(diff)
        a = diff

    return b << shift


cdef _py_gcd(ullong a, ullong b):
    return _c_gcd(a, b) if b else a


cdef ullong _c_gcd_euclid(ullong a, ullong b):
    if not b:
        return a
    if a <= <ullong>INT_MAX*2+1 and b <= <ullong>INT_MAX*2+1:
        return _euclid_gcd[uint](<uint> a, <uint> b)
    elif a <= <ullong>LONG_MAX*2+1 and b <= <ullong>LONG_MAX*2+1:
        return _euclid_gcd[ulong](<ulong> a, <ulong> b)
    else:
        return _euclid_gcd[ullong](a, b)


cdef ullong _c_gcd_binary(ullong a, ullong b):
    if not b:
        return a
    if HAS_FAST_CTZ_uint and a <= <ullong>INT_MAX*2+1 and b <= <ullong>INT_MAX*2+1:
        return _binary_gcd[uint](<uint> a, <uint> b)
    elif HAS_FAST_CTZ_ulong and a <= <ullong>LONG_MAX*2+1 and b <= <ullong>LONG_MAX*2+1:
        return _binary_gcd[ulong](<ulong> a, <ulong> b)
    elif HAS_FAST_CTZ_ullong:
        return _binary_gcd[ullong](a, b)
    else:
        return _c_gcd_euclid(a, b)


cdef ullong _c_gcd_hybrid2(ullong a, ullong b):
    if not b:
        return a
    if HAS_FAST_CTZ_uint and a <= <ullong>INT_MAX*2+1 and b <= <ullong>INT_MAX*2+1:
        return _hybrid_binary_gcd2[uint](<uint> a, <uint> b)
    elif HAS_FAST_CTZ_ulong and a <= <ullong>LONG_MAX*2+1 and b <= <ullong>LONG_MAX*2+1:
        return _hybrid_binary_gcd2[ulong](<ulong> a, <ulong> b)
    elif HAS_FAST_CTZ_ullong:
        return _hybrid_binary_gcd2[ullong](a, b)
    else:
        return _c_gcd_euclid(a, b)


cdef ullong _c_gcd_hybrid(ullong a, ullong b):
    if not b:
        return a
    if HAS_FAST_CTZ_uint and a <= <ullong>INT_MAX and b <= <ullong>INT_MAX:
        return _hybrid_binary_gcd[uint](<uint> a, <uint> b)
    elif HAS_FAST_CTZ_ulong and a <= <ullong>LONG_MAX and b <= <ullong>LONG_MAX:
        return _hybrid_binary_gcd[ulong](<ulong> a, <ulong> b)
    elif HAS_FAST_CTZ_ullong:
        return _hybrid_binary_gcd[ullong](a, b)
    else:
        return _c_gcd_euclid(a, b)


ctypedef ullong (*fast_cgcd)(ullong a, ullong b)

cdef fast_cgcd _c_gcd = NULL
cdef fast_cgcd _c_gcd_best_hybrid = NULL


def use_gcd_impl(name):
    """Change the internal GCD implementation.

    'name' is one of: 'euclid', 'binary', 'hybrid'
    """
    if name not in ('euclid', 'binary', 'hybrid'):
        raise ValueError(f"{name!r} is not one of: 'euclid', 'binary', 'hybrid'")

    global _c_gcd, GCD_IMPL

    if name == 'euclid':
        _c_gcd = _c_gcd_euclid
    elif name == 'binary':
        _c_gcd = _c_gcd_binary
    else:
        _c_gcd = _c_gcd_best_hybrid
    GCD_IMPL = name


cdef double _measure_gcd_performance(fast_cgcd func):
    from time import perf_counter as timer
    cdef int i, j, e

    t = timer()

    # Fibonacci numbers:
    cdef unsigned long long a = 1, b = 1, c = 2

    for i in range(1, 91):
        a, b, c = b, a+b, b+c
        if a & 3 == 0:
            func(a, b)
            func(a, c)

    # Close numbers:
    for i in range(1, 20):
        func(209865, 209797)

    # 'Parsed' digits with 10^n denominator:
    cdef unsigned long long num = 3
    cdef unsigned long long denom = 1
    cdef unsigned long long n

    for i in range(1, 18):
        num = 10 * num + (i & 7)
        denom *= 10
        n = num
        for e in range(3):
            n *= 10
            for j in range(8):
                func(n + j, denom)

    t = timer() - t

    return t


cdef int _tune_cgcd():
    # Find fastest GCD implementation for this machine.
    t_binary = _measure_gcd_performance(_c_gcd_binary)
    t_euclid = _measure_gcd_performance(_c_gcd_euclid)
    t_hybrid = _measure_gcd_performance(_c_gcd_hybrid)
    t_hybrid2 = _measure_gcd_performance(_c_gcd_hybrid2)

    name = 'euclid'
    best_time = t_euclid

    if t_binary < best_time:
        name = 'binary'
        best_time = t_binary

    if t_hybrid < best_time:
        name = 'hybrid'
        best_time = t_hybrid

    if t_hybrid2 < best_time:
        name = 'hybrid'
        best_time = t_hybrid2

    global _c_gcd_best_hybrid
    _c_gcd_best_hybrid = _c_gcd_hybrid if t_hybrid < t_hybrid2 else _c_gcd_hybrid2

    #print(f"E: {t_euclid * 1_000_000:.2f}, B: {t_binary * 1_000_000:.2f}, H1: {t_hybrid * 1_000_000:.2f}, H2: {t_hybrid2 * 1_000_000:.2f}")
    use_gcd_impl(name)

    return 0

_tune_cgcd()


cdef _gcd_fallback(a: int, b: int):
    """Fallback GCD implementation if _PyLong_GCD() is not available.
    """
    # Try doing the computation in C space.  If the numbers are too
    # large at the beginning, do object calculations until they are small enough.
    cdef ullong au, bu
    cdef long long ai, bi

    if HAS_ISLONGLONG:
        if PyLong_IsLongLong(a) and PyLong_IsLongLong(b):
            ai, bi = a, b
            au = _abs(ai)
            bu = _abs(bi)
            return _py_gcd(au, bu)
    else:
        # Optimistically try to switch to C space.
        try:
            ai, bi = a, b
        except OverflowError:
            pass
        else:
            au = _abs(ai)
            bu = _abs(bi)
            return _py_gcd(au, bu)

    # Do object calculation until we reach the C space limit.
    a = abs(a)
    b = abs(b)
    while b > PY_MAX_ULONGLONG:
        a, b = b, a%b
    while b and a > PY_MAX_ULONGLONG:
        a, b = b, a%b
    if not b:
        return a
    return _py_gcd(a, b)


# Constants related to the hash implementation;  hash(x) is based
# on the reduction of x modulo the prime _PyHASH_MODULUS.

from sys import hash_info

cdef Py_hash_t _PyHASH_MODULUS = hash_info.modulus

# Value to be used for rationals that reduce to infinity modulo
# _PyHASH_MODULUS.
cdef Py_hash_t _PyHASH_INF = hash_info.inf

del hash_info


# Helpers for formatting

cdef _round_to_exponent(n, d, exponent, bint no_neg_zero=False):
    """Round a rational number to the nearest multiple of a given power of 10.

    Rounds the rational number n/d to the nearest integer multiple of
    10**exponent, rounding to the nearest even integer multiple in the case of
    a tie. Returns a pair (sign: bool, significand: int) representing the
    rounded value (-1)**sign * significand * 10**exponent.

    If no_neg_zero is true, then the returned sign will always be False when
    the significand is zero. Otherwise, the sign reflects the sign of the
    input.

    d must be positive, but n and d need not be relatively prime.
    """
    if exponent >= 0:
        d *= 10**exponent
    else:
        n *= 10**-exponent

    # The divmod quotient is correct for round-ties-towards-positive-infinity;
    # In the case of a tie, we zero out the least significant bit of q.
    q, r = divmod(n + (d >> 1), d)
    if r == 0 and d & 1 == 0:
        q &= -2

    cdef bint sign = q < 0 if no_neg_zero else n < 0
    return sign, abs(q)


cdef _round_to_figures(n, d, Py_ssize_t figures):
    """Round a rational number to a given number of significant figures.

    Rounds the rational number n/d to the given number of significant figures
    using the round-ties-to-even rule, and returns a triple
    (sign: bool, significand: int, exponent: int) representing the rounded
    value (-1)**sign * significand * 10**exponent.

    In the special case where n = 0, returns a significand of zero and
    an exponent of 1 - figures, for compatibility with formatting.
    Otherwise, the returned significand satisfies
    10**(figures - 1) <= significand < 10**figures.

    d must be positive, but n and d need not be relatively prime.
    figures must be positive.
    """
    # Special case for n == 0.
    if n == 0:
        return False, 0, 1 - figures

    cdef bint sign

    # Find integer m satisfying 10**(m - 1) <= abs(n)/d <= 10**m. (If abs(n)/d
    # is a power of 10, either of the two possible values for m is fine.)
    str_n, str_d = str(abs(n)), str(d)
    cdef Py_ssize_t m = len(str_n) - len(str_d) + (str_d <= str_n)

    # Round to a multiple of 10**(m - figures). The significand we get
    # satisfies 10**(figures - 1) <= significand <= 10**figures.
    exponent = m - figures
    sign, significand = _round_to_exponent(n, d, exponent)

    # Adjust in the case where significand == 10**figures, to ensure that
    # 10**(figures - 1) <= significand < 10**figures.
    if len(str(significand)) == figures + 1:
        significand //= 10
        exponent += 1

    return sign, significand, exponent


# Pattern for matching non-float-style format specifications.
cdef object _GENERAL_FORMAT_SPECIFICATION_MATCHER = re.compile(r"""
    (?:
        (?P<fill>.)?
        (?P<align>[<>=^])
    )?
    (?P<sign>[-+ ]?)
    # Alt flag forces a slash and denominator in the output, even for
    # integer-valued Fraction objects.
    (?P<alt>\#)?
    # We don't implement the zeropad flag since there's no single obvious way
    # to interpret it.
    (?P<minimumwidth>0|[1-9][0-9]*)?
    (?P<thousands_sep>[,_])?
    $
""", re.DOTALL | re.VERBOSE).match


# Pattern for matching float-style format specifications;
# supports 'e', 'E', 'f', 'F', 'g', 'G' and '%' presentation types.
cdef object _FLOAT_FORMAT_SPECIFICATION_MATCHER = re.compile(r"""
    (?:
        (?P<fill>.)?
        (?P<align>[<>=^])
    )?
    (?P<sign>[-+ ]?)
    (?P<no_neg_zero>z)?
    (?P<alt>\#)?
    # A '0' that's *not* followed by another digit is parsed as a minimum width
    # rather than a zeropad flag.
    (?P<zeropad>0(?=[0-9]))?
    (?P<minimumwidth>[0-9]+)?
    (?P<thousands_sep>[,_])?
    (?:\.(?P<precision>[0-9]+))?
    (?P<presentation_type>[eEfFgG%])
    $
""", re.DOTALL | re.VERBOSE).match

cdef object NOINIT = object()


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

    def __cinit__(self, numerator=0, denominator=None):
        self._hash = -1
        if numerator is NOINIT:
            return  # fast-path for external initialisation

        cdef bint _normalize = True
        if denominator is None:
            if type(numerator) is int:
                self._numerator = numerator
                self._denominator = 1
                return

            elif type(numerator) is float:
                # Exact conversion
                self._numerator, self._denominator = numerator.as_integer_ratio()
                return

            elif type(numerator) is Fraction:
                self._numerator = (<Fraction>numerator)._numerator
                self._denominator = (<Fraction>numerator)._denominator
                return

            elif isinstance(numerator, unicode):
                numerator, denominator, is_normalised = _parse_fraction[unicode](
                    <unicode>numerator, len(<unicode>numerator), numerator)
                if is_normalised:
                    _normalize = False
                # fall through to normalisation below

            elif isinstance(numerator, float) or (
                    not isinstance(numerator, type) and hasattr(numerator, 'as_integer_ratio')):
                # Exact conversion
                self._numerator, self._denominator = numerator.as_integer_ratio()
                return

            elif isinstance(numerator, (Fraction, Rational)):
                self._numerator = numerator.numerator
                self._denominator = numerator.denominator
                return

            elif isinstance(numerator, Decimal):
                self._numerator, self._denominator = numerator.as_integer_ratio()
                return

            else:
                raise TypeError("argument should be a string or a Rational "
                                "instance or have the as_integer_ratio() method")

        elif type(numerator) is int is type(denominator):
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
            raise ZeroDivisionError(f'Fraction({numerator}, 0)')
        if _normalize:
            if not isinstance(numerator, int):
                numerator = int(numerator)
            if not isinstance(denominator, int):
                denominator = int(denominator)
            g = _igcd(numerator, denominator)
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
    def from_number(cls, number):
        """Converts a finite real number to a rational number, exactly.

        Beware that Fraction.from_number(0.3) != Fraction(3, 10).

        """
        if type(number) is Fraction:
            return _fraction_from_coprime_ints((<Fraction> number)._numerator, (<Fraction> number)._denominator, cls)

        elif isinstance(number, int):
            return _fraction_from_coprime_ints(number, 1, cls)

        elif isinstance(number, float):
            n, d = number.as_integer_ratio()
            return _fraction_from_coprime_ints(n, d, cls)

        elif isinstance(number, Rational):
            return _fraction_from_coprime_ints(number.numerator, number.denominator, cls)

        elif not isinstance(number, type) and hasattr(number, 'as_integer_ratio'):
            n, d = number.as_integer_ratio()
            return _fraction_from_coprime_ints(n, d, cls)

        else:
            raise TypeError("argument should be a Rational instance or "
                            "have the as_integer_ratio() method")

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

        if isinstance(f, Integral):
            return cls(f)
        elif not isinstance(f, float):
            raise TypeError(f"{cls.__name__}.from_float() only takes floats, not {f!r} ({type(f).__name__})")
        if math.isinf(f):
            raise OverflowError(f"Cannot convert {f!r} to {cls.__name__}.")
        raise ValueError(f"Cannot convert {f!r} to {cls.__name__}.")

    @classmethod
    def from_decimal(cls, dec):
        """Converts a finite Decimal instance to a rational number, exactly."""
        cdef Py_ssize_t exp
        if isinstance(dec, Integral):
            dec = Decimal(int(dec))
        elif not isinstance(dec, Decimal):
            raise TypeError(
                f"{cls.__name__}.from_decimal() only takes Decimals, not {dec!r} ({type(dec).__name__})")
        if dec.is_infinite():
            raise OverflowError(f"Cannot convert {dec} to {cls.__name__}.")
        if dec.is_nan():
            raise ValueError(f"Cannot convert {dec} to {cls.__name__}.")

        num, denom = dec.as_integer_ratio()
        return _fraction_from_coprime_ints(num, denom, cls)

    def is_integer(self):
        """Return True if the Fraction is an integer."""
        return self._denominator == 1

    def as_integer_ratio(self):
        """Return a pair of integers, whose ratio is equal to the original Fraction.

        The ratio is in lowest terms and has a positive denominator.
        """
        return (self._numerator, self._denominator)

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

        # Determine which of the candidates (p0+k*p1)/(q0+k*q1) and p1/q1 is
        # closer to self. The distance between them is 1/(q1*(q0+k*q1)), while
        # the distance from p1/q1 to self is d/(q1*self._denominator). So we
        # need to compare 2*(q0+k*q1) with self._denominator/d.
        if 2*d*(q0+k*q1) <= self._denominator:
            return _fraction_from_coprime_ints(p1, q1)
        else:
            return _fraction_from_coprime_ints(p0+k*p1, q0+k*q1)

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    def __repr__(self):
        """repr(self)"""
        return '%s(%s, %s)' % (self.__class__.__name__,
                               self._numerator, self._denominator)

    def __str__(self):
        """str(self)"""
        if self._denominator == 1:
            return str(self._numerator)
        else:
            return f'{self._numerator}/{self._denominator}'

    @cython.final
    cdef _format_general(self, dict match):
        """Helper method for __format__.

        Handles fill, alignment, signs, and thousands separators in the
        case of no presentation type.
        """
        # Validate and parse the format specifier.
        fill = match["fill"] or " "
        cdef Py_UCS4 align = ord(match["align"] or ">")
        pos_sign = "" if match["sign"] == "-" else match["sign"]
        cdef bint alternate_form = match["alt"]
        cdef Py_ssize_t minimumwidth = int(match["minimumwidth"] or "0")
        thousands_sep = match["thousands_sep"] or ''

        cdef Py_ssize_t first_pos  # Py2/3.5-only

        # Determine the body and sign representation.
        n, d = self._numerator, self._denominator
        if d > 1 or alternate_form:
            body = f"{abs(n):{thousands_sep}}/{d:{thousands_sep}}"
        else:
            body = f"{abs(n):{thousands_sep}}"
        sign = '-' if n < 0 else pos_sign

        # Pad with fill character if necessary and return.
        padding = fill * (minimumwidth - len(sign) - len(body))
        if align == u">":
            return padding + sign + body
        elif align == u"<":
            return sign + body + padding
        elif align == u"^":
            half = len(padding) // 2
            return padding[:half] + sign + body + padding[half:]
        else:  # align == u"="
            return sign + padding + body

    @cython.final
    cdef _format_float_style(self, dict match):
        """Helper method for __format__; handles float presentation types."""
        fill = match["fill"] or " "
        align = match["align"] or ">"
        pos_sign = "" if match["sign"] == "-" else match["sign"]
        cdef bint no_neg_zero = match["no_neg_zero"]
        cdef bint alternate_form = match["alt"]
        cdef bint zeropad = match["zeropad"]
        cdef Py_ssize_t minimumwidth = int(match["minimumwidth"] or "0")
        thousands_sep = match["thousands_sep"]
        cdef Py_ssize_t precision = int(match["precision"] or "6")
        cdef Py_UCS4 presentation_type = ord(match["presentation_type"])
        cdef bint trim_zeros = presentation_type in u"gG" and not alternate_form
        cdef bint trim_point = not alternate_form
        exponent_indicator = "E" if presentation_type in u"EFG" else "e"

        cdef bint negative, scientific
        cdef Py_ssize_t exponent, figures

        # Round to get the digits we need, figure out where to place the point,
        # and decide whether to use scientific notation. 'point_pos' is the
        # relative to the _end_ of the digit string: that is, it's the number
        # of digits that should follow the point.
        if presentation_type in u"fF%":
            exponent = -precision
            if presentation_type == u"%":
                exponent -= 2
            negative, significand = _round_to_exponent(
                self._numerator, self._denominator, exponent, no_neg_zero)
            scientific = False
            point_pos = precision
        else:  # presentation_type in "eEgG"
            figures = (
                max(precision, 1)
                if presentation_type in u"gG"
                else precision + 1
            )
            negative, significand, exponent = _round_to_figures(
                self._numerator, self._denominator, figures)
            scientific = (
                presentation_type in u"eE"
                or exponent > 0
                or exponent + figures <= -4
            )
            point_pos = figures - 1 if scientific else -exponent

        # Get the suffix - the part following the digits, if any.
        if presentation_type == u"%":
            suffix = "%"
        elif scientific:
            #suffix = f"{exponent_indicator}{exponent + point_pos:+03d}"
            suffix = "%s%+03d" % (exponent_indicator, exponent + point_pos)
        else:
            suffix = ""

        # String of output digits, padded sufficiently with zeros on the left
        # so that we'll have at least one digit before the decimal point.
        digits = f"{significand:0{point_pos + 1}d}"

        # Before padding, the output has the form f"{sign}{leading}{trailing}",
        # where `leading` includes thousands separators if necessary and
        # `trailing` includes the decimal separator where appropriate.
        sign = "-" if negative else pos_sign
        leading = digits[: len(digits) - point_pos]
        frac_part = digits[len(digits) - point_pos :]
        if trim_zeros:
            frac_part = frac_part.rstrip("0")
        separator = "" if trim_point and not frac_part else "."
        trailing = separator + frac_part + suffix

        # Do zero padding if required.
        if zeropad:
            min_leading = minimumwidth - len(sign) - len(trailing)
            # When adding thousands separators, they'll be added to the
            # zero-padded portion too, so we need to compensate.
            leading = leading.zfill(
                3 * min_leading // 4 + 1 if thousands_sep else min_leading
            )

        # Insert thousands separators if required.
        if thousands_sep:
            first_pos = 1 + (len(leading) - 1) % 3
            leading = leading[:first_pos] + "".join([
                thousands_sep + leading[pos : pos + 3]
                for pos in range(first_pos, len(leading), 3)
            ])

        # We now have a sign and a body. Pad with fill character if necessary
        # and return.
        body = leading + trailing
        padding = fill * (minimumwidth - len(sign) - len(body))
        if align == ">":
            return padding + sign + body
        elif align == "<":
            return sign + body + padding
        elif align == "^":
            half = len(padding) // 2
            return padding[:half] + sign + body + padding[half:]
        else:  # align == "="
            return sign + padding + body

    def __format__(self, format_spec, /):
        """Format this fraction according to the given format specification."""

        if match := _GENERAL_FORMAT_SPECIFICATION_MATCHER(format_spec):
            return self._format_general(match.groupdict())

        if match := _FLOAT_FORMAT_SPECIFICATION_MATCHER(format_spec):
            # Refuse the temptation to guess if both alignment _and_
            # zero padding are specified.
            match_groups = match.groupdict()
            if match_groups["align"] is None or match_groups["zeropad"] is None:
                return self._format_float_style(match_groups)

        raise ValueError(
            f"Invalid format specifier {format_spec!r} "
            f"for object of type {type(self).__name__!r}"
        )

    def __add__(a, b):
        """a + b"""
        return forward(a, b, _add, _math_op_add)

    def __radd__(b, a):
        """a + b"""
        return reverse(a, b, _add, _math_op_add)

    def __sub__(a, b):
        """a - b"""
        return forward(a, b, _sub, _math_op_sub)

    def __rsub__(b, a):
        """a - b"""
        return reverse(a, b, _sub, _math_op_sub)

    def __mul__(a, b):
        """a * b"""
        return forward(a, b, _mul, _math_op_mul)

    def __rmul__(b, a):
        """a * b"""
        return reverse(a, b, _mul, _math_op_mul)

    def __div__(a, b):
        """a / b"""
        return forward(a, b, _div, _math_op_div)

    def __rdiv__(b, a):
        """a / b"""
        return reverse(a, b, _div, _math_op_div)

    def __truediv__(a, b):
        """a / b"""
        return forward(a, b, _div, _math_op_truediv)

    def __rtruediv__(b, a):
        """a / b"""
        return reverse(a, b, _div, _math_op_truediv)

    def __floordiv__(a, b):
        """a // b"""
        return forward(a, b, _floordiv, _math_op_floordiv, handle_complex=False)

    def __rfloordiv__(b, a):
        """a // b"""
        return reverse(a, b, _floordiv, _math_op_floordiv, handle_complex=False)

    def __mod__(a, b):
        """a % b"""
        return forward(a, b, _mod, _math_op_mod, handle_complex=False)

    def __rmod__(b, a):
        """a % b"""
        return reverse(a, b, _mod, _math_op_mod, handle_complex=False)

    def __divmod__(a, b):
        """divmod(self, other): The pair (self // other, self % other).

        Sometimes this can be computed faster than the pair of
        operations.
        """
        return forward(a, b, _divmod, _math_op_divmod, handle_complex=False)

    def __rdivmod__(b, a):
        """divmod(self, other): The pair (self // other, self % other).

        Sometimes this can be computed faster than the pair of
        operations.
        """
        return reverse(a, b, _divmod, _math_op_divmod, handle_complex=False)

    def __pow__(a, b, x):
        """a ** b

        If b is not an integer, the result will be a float or complex
        since roots are generally irrational. If b is an integer, the
        result will be rational.
        """
        if x is not None:
            return NotImplemented

        if isinstance(b, int):
            return _pow(a.numerator, a.denominator, b, 1)
        elif isinstance(b, (Fraction, Rational)):
            return _pow(a.numerator, a.denominator, b.numerator, b.denominator)
        elif isinstance(b, (float, complex)):
            return (a.numerator / a.denominator) ** b
        else:
            return NotImplemented

    def __rpow__(b, a, x):
        """a ** b

        If b is not an integer, the result will be a float or complex
        since roots are generally irrational. If b is an integer, the
        result will be rational.
        """
        if x is not None:
            return NotImplemented

        bn, bd = b.numerator, b.denominator
        if bd == 1 and bn >= 0:
            # If a is an int, keep it that way if possible.
            return a ** bn

        if isinstance(a, int):
            return _pow(a, 1, bn, bd)
        if isinstance(a, (Fraction, Rational)):
            return _pow(a.numerator, a.denominator, bn, bd)

        if bd == 1:
            return a ** bn

        return a ** (bn / bd)

    def __pos__(a):
        """+a: Coerces a subclass instance to Fraction"""
        if type(a) is Fraction:
            return a
        return _fraction_from_coprime_ints(a._numerator, a._denominator)

    def __neg__(a):
        """-a"""
        return _fraction_from_coprime_ints(-a._numerator, a._denominator)

    def __abs__(a):
        """abs(a)"""
        return _fraction_from_coprime_ints(abs(a._numerator), a._denominator)

    def __int__(a):
        """int(a)"""
        if a._numerator < 0:
            return _operator_index(-(-a._numerator // a._denominator))
        else:
            return _operator_index(a._numerator // a._denominator)

    def __trunc__(a):
        """math.trunc(a)"""
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
        shift = pow10(abs(<long long>ndigits))
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
    @property
    def imag(self):
        "Real numbers have no imaginary component."
        return 0

    def conjugate(self):
        """Conjugate is a no-op for Reals."""
        return +self

    def __hash__(self):
        """hash(self)"""
        if self._hash != -1:
            return self._hash

        cdef Py_hash_t result

        # In order to make sure that the hash of a Fraction agrees
        # with the hash of a numerically equal integer, float or
        # Decimal instance, we follow the rules for numeric hashes
        # outlined in the documentation.  (See library docs, 'Built-in
        # Types').

        try:
            dinv = pow(self._denominator, -1, _PyHASH_MODULUS)
        except ValueError:
            # ValueError means there is no modular inverse.
            result = _PyHASH_INF
        else:
            # The general algorithm now specifies that the absolute value of
            # the hash is
            #    (|N| * dinv) % P
            # where N is self._numerator and P is _PyHASH_MODULUS.  That's
            # optimized here in two ways:  first, for a non-negative int i,
            # hash(i) == i % P, but the int hash implementation doesn't need
            # to divide, and is faster than doing % P explicitly.  So we do
            #    hash(|N| * dinv)
            # instead.  Second, N is unbounded, so its product with dinv may
            # be arbitrarily expensive to compute.  The final answer is the
            # same if we use the bounded |N| % P instead, which can again
            # be done with an int hash() call.  If 0 <= i < P, hash(i) == i,
            # so this nested hash() call wastes a bit of time making a
            # redundant copy when |N| < P, but can save an arbitrarily large
            # amount of computation for large |N|.
            result = hash(hash(abs(self._numerator)) * dinv)

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
        else:
            a, b = b, a
            if op == Py_EQ:
                return (<Fraction>a)._eq(b)
            elif op == Py_NE:
                result = (<Fraction>a)._eq(b)
                return NotImplemented if result is NotImplemented else not result
            elif op == Py_LT:
                op = Py_GE
            elif op == Py_GT:
                op = Py_LE
            elif op == Py_LE:
                op = Py_GT
            elif op == Py_GE:
                op = Py_LT
            else:
                return NotImplemented
        return (<Fraction>a)._richcmp(b, op)

    @cython.final
    cdef _eq(a, b):
        if type(b) is int:
            return a._numerator == b and a._denominator == 1
        if type(b) is Fraction:
            return (a._numerator == (<Fraction>b)._numerator and
                    a._denominator == (<Fraction>b)._denominator)
        if isinstance(b, Rational):
            return (a._numerator == b.numerator and
                    a._denominator == b.denominator)
        if isinstance(b, Complex) and b.imag == 0:
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
    cdef _richcmp(self, other, int op):
        """Helper for comparison operators, for internal use only.

        Implement comparison between a Rational instance `self`, and
        either another Rational instance or a float `other`.  If
        `other` is not a Rational instance or a float, return
        NotImplemented. `op` should be one of the six standard
        comparison operators.

        """
        # convert other to a Rational instance where reasonable.
        if isinstance(other, int):
            a = self._numerator
            b = self._denominator * other
        elif type(other) is Fraction:
            a = self._numerator * (<Fraction>other)._denominator
            b = self._denominator * (<Fraction>other)._numerator
        elif isinstance(other, float):
            if math.isnan(other) or math.isinf(other):
                a, b = 0.0, other  # Comparison to 0.0 is just as good as any.
            else:
                return self._richcmp(self.from_float(other), op)
        elif isinstance(other, (Fraction, Rational)):
            a = self._numerator * other.denominator
            b = self._denominator * other.numerator
        else:
            # comparisons with complex should raise a TypeError, for consistency
            # with int<->complex, float<->complex, and complex<->complex comparisons.
            return NotImplemented

        if op == Py_LT:
            return a < b
        elif op == Py_GT:
            return a > b
        elif op == Py_LE:
            return a <= b
        elif op == Py_GE:
            return a >= b
        else:
            return NotImplemented

    def __bool__(self):
        """a != 0"""
        # Use bool() because (a._numerator != 0) can return an
        # object which is not a bool.
        # See https://bugs.python.org/issue39274
        return bool(self._numerator)

    # support for pickling, copy, and deepcopy

    def __reduce__(self):
        return (type(self), (self._numerator, self._denominator))

    def __copy__(self):
        if type(self) is Fraction:
            return self     # I'm immutable; therefore I am my own clone
        return type(self)(self._numerator, self._denominator)

    def __deepcopy__(self, memo):
        if type(self) is Fraction:
            return self     # My components are also immutable
        return type(self)(self._numerator, self._denominator)


# Register with Python's numerical tower.
Rational.register(Fraction)


cdef _fraction_from_coprime_ints(numerator, denominator, cls=None):
    """Convert a pair of ints to a rational number, for internal use.

    The ratio of integers should be in lowest terms and the denominator
    should be positive.
    """
    cdef Fraction obj
    if cls is None or cls is Fraction:
        obj = Fraction.__new__(Fraction, NOINIT, NOINIT)
    else:
        obj = cls.__new__(cls)
    obj._numerator = numerator
    obj._denominator = denominator
    return obj


cdef _pow(an, ad, bn, bd):
    if bd == 1:
        # power = bn
        if bn >= 0:
            return _fraction_from_coprime_ints(
                an ** bn,
                ad ** bn)
        elif an > 0:
            return _fraction_from_coprime_ints(
                ad ** -bn,
                an ** -bn)
        elif an == 0:
            raise ZeroDivisionError(f'Fraction({ad ** -bn}, 0)')
        else:
            return _fraction_from_coprime_ints(
                (-ad) ** -bn,
                (-an) ** -bn)
    else:
        # A fractional power will generally produce an
        # irrational number.
        return (an / ad) ** (bn / bd)


cdef _as_float(numerator, denominator):
    return numerator / denominator


# Rational arithmetic algorithms: Knuth, TAOCP, Volume 2, 4.5.1.
#
# Assume input fractions a and b are normalized.
#
# 1) Consider addition/subtraction.
#
# Let g = gcd(da, db). Then
#
#              na   nb    na*db ± nb*da
#     a ± b == -- ± -- == ------------- ==
#              da   db        da*db
#
#              na*(db//g) ± nb*(da//g)    t
#           == ----------------------- == -
#                      (da*db)//g         d
#
# Now, if g > 1, we're working with smaller integers.
#
# Note, that t, (da//g) and (db//g) are pairwise coprime.
#
# Indeed, (da//g) and (db//g) share no common factors (they were
# removed) and da is coprime with na (since input fractions are
# normalized), hence (da//g) and na are coprime.  By symmetry,
# (db//g) and nb are coprime too.  Then,
#
#     gcd(t, da//g) == gcd(na*(db//g), da//g) == 1
#     gcd(t, db//g) == gcd(nb*(da//g), db//g) == 1
#
# Above allows us optimize reduction of the result to lowest
# terms.  Indeed,
#
#     g2 = gcd(t, d) == gcd(t, (da//g)*(db//g)*g) == gcd(t, g)
#
#                       t//g2                   t//g2
#     a ± b == ----------------------- == ----------------
#              (da//g)*(db//g)*(g//g2)    (da//g)*(db//g2)
#
# is a normalized fraction.  This is useful because the unnormalized
# denominator d could be much larger than g.
#
# We should special-case g == 1 (and g2 == 1), since 60.8% of
# randomly-chosen integers are coprime:
# https://en.wikipedia.org/wiki/Coprime_integers#Probability_of_coprimality
# Note, that g2 == 1 always for fractions, obtained from floats: here
# g is a power of 2 and the unnormalized numerator t is an odd integer.
#
# 2) Consider multiplication
#
# Let g1 = gcd(na, db) and g2 = gcd(nb, da), then
#
#            na*nb    na*nb    (na//g1)*(nb//g2)
#     a*b == ----- == ----- == -----------------
#            da*db    db*da    (db//g1)*(da//g2)
#
# Note, that after divisions we're multiplying smaller integers.
#
# Also, the resulting fraction is normalized, because each of
# two factors in the numerator is coprime to each of the two factors
# in the denominator.
#
# Indeed, pick (na//g1).  It's coprime with (da//g2), because input
# fractions are normalized.  It's also coprime with (db//g1), because
# common factors are removed by g1 == gcd(na, db).
#
# As for addition/subtraction, we should special-case g1 == 1
# and g2 == 1 for same reason.  That happens also for multiplying
# rationals, obtained from floats.

cdef _add(na, da, nb, db):
    """a + b"""
    # return Fraction(na * db + nb * da, da * db)
    g = _gcd(da, db)
    if g == 1:
        return _fraction_from_coprime_ints(na * db + da * nb, da * db)
    s = da // g
    t = na * (db // g) + nb * s
    g2 = _gcd(t, g)
    if g2 == 1:
        return _fraction_from_coprime_ints(t, s * db)
    return _fraction_from_coprime_ints(t // g2, s * (db // g2))

cdef _sub(na, da, nb, db):
    """a - b"""
    # return Fraction(na * db - nb * da, da * db)
    g = _gcd(da, db)
    if g == 1:
        return _fraction_from_coprime_ints(na * db - da * nb, da * db)
    s = da // g
    t = na * (db // g) - nb * s
    g2 = _gcd(t, g)
    if g2 == 1:
        return _fraction_from_coprime_ints(t, s * db)
    return _fraction_from_coprime_ints(t // g2, s * (db // g2))

cdef _mul(na, da, nb, db):
    """a * b"""
    # return Fraction(na * nb, da * db)
    g1 = _gcd(na, db)
    if g1 > 1:
        na //= g1
        db //= g1
    g2 = _gcd(nb, da)
    if g2 > 1:
        nb //= g2
        da //= g2
    return _fraction_from_coprime_ints(na * nb, db * da)

cdef _div(na, da, nb, db):
    """a / b"""
    # return Fraction(na * db, da * nb)
    # Same as _mul(), with inversed b.
    if nb == 0:
        raise ZeroDivisionError(f'Fraction({db}, 0)')
    g1 = _gcd(na, nb)
    if g1 > 1:
        na //= g1
        nb //= g1
    g2 = _gcd(db, da)
    if g2 > 1:
        da //= g2
        db //= g2
    n, d = na * db, nb * da
    if d < 0:
        n, d = -n, -d
    return _fraction_from_coprime_ints(n, d)

cdef _floordiv(an, ad, bn, bd):
    """a // b -> int"""
    return (an * bd) // (bn * ad)

cdef _divmod(an, ad, bn, bd):
    div, n_mod = divmod(an * bd, ad * bn)
    return div, Fraction(n_mod, ad * bd)

cdef _mod(an, ad, bn, bd):
    return Fraction((an * bd) % (bn * ad), ad * bd)


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

cdef:
    _math_op_add = operator.add
    _math_op_sub = operator.sub
    _math_op_mul = operator.mul
    _math_op_div = getattr(operator, 'div', operator.truediv)  # Py2/3
    _math_op_truediv = operator.truediv
    _math_op_floordiv = operator.floordiv
    _math_op_mod = operator.mod
    _math_op_divmod = divmod


ctypedef object (*math_func)(an, ad, bn, bd)


cdef forward(a, b, math_func monomorphic_operator, pyoperator, handle_complex=True):
    an, ad = (<Fraction>a)._numerator, (<Fraction>a)._denominator
    if type(b) is Fraction:
        return monomorphic_operator(an, ad, (<Fraction>b)._numerator, (<Fraction>b)._denominator)
    elif isinstance(b, int):
        return monomorphic_operator(an, ad, b, 1)
    elif isinstance(b, Fraction):
        return monomorphic_operator(an, ad, b.numerator, b.denominator)
    elif isinstance(b, float):
        return pyoperator(_as_float(an, ad), b)
    elif handle_complex and isinstance(b, complex):
        return pyoperator(float(a), b)
    else:
        return NotImplemented


cdef reverse(a, b, math_func monomorphic_operator, pyoperator, handle_complex=True):
    bn, bd = (<Fraction>b)._numerator, (<Fraction>b)._denominator
    if isinstance(a, int):
        return monomorphic_operator(a, 1, bn, bd)
    elif isinstance(a, Rational):
        return monomorphic_operator(a.numerator, a.denominator, bn, bd)
    elif isinstance(a, Real):
        return pyoperator(float(a), _as_float(bn, bd))
    elif handle_complex and isinstance(a, Complex):
        return pyoperator(complex(a), float(b))
    else:
        return NotImplemented


ctypedef char* charptr

ctypedef fused AnyString:
    unicode
    charptr


cdef enum ParserState:
    BEGIN_SPACE          # '\s'*     ->  (BEGIN_SIGN, SMALL_NUM, START_DECIMAL_DOT)
    BEGIN_SIGN           # [+-]      ->  (SMALL_NUM, SMALL_DECIMAL_DOT)
    SMALL_NUM            # [0-9]+    ->  (SMALL_NUM, SMALL_NUM_US, NUM, NUM_SPACE, SMALL_DECIMAL_DOT, EXP_E, DENOM_START)
    SMALL_NUM_US         # '_'       ->  (SMALL_NUM, NUM)
    NUM                  # [0-9]+    ->  (NUM, NUM_US, NUM_SPACE, DECIMAL_DOT, EXP_E, DENOM_START)
    NUM_US               # '_'       ->  (NUM)
    NUM_SPACE            # '\s'+     ->  (DENOM_START)

    # 1) floating point syntax
    START_DECIMAL_DOT    # '.'       ->  (SMALL_DECIMAL)
    SMALL_DECIMAL_DOT    # '.'       ->  (SMALL_DECIMAL, EXP_E, SMALL_END_SPACE)
    DECIMAL_DOT          # '.'       ->  (DECIMAL, EXP_E, END_SPACE)
    SMALL_DECIMAL        # [0-9]+    ->  (SMALL_DECIMAL, SMALL_DECIMAL_US, DECIMAL, EXP_E, SMALL_END_SPACE)
    SMALL_DECIMAL_US     # '_'       ->  (SMALL_DECIMAL, DECIMAL)
    DECIMAL              # [0-9]+    ->  (DECIMAL, DECIMAL_US, EXP_E, END_SPACE)
    DECIMAL_US           # '_'       ->  (DECIMAL)
    EXP_E                # [eE]      ->  (EXP_SIGN, EXP)
    EXP_SIGN             # [+-]      ->  (EXP)
    EXP                  # [0-9]+    ->  (EXP_US, END_SPACE)
    EXP_US               # '_'       ->  (EXP)
    END_SPACE            # '\s'+
    SMALL_END_SPACE      # '\s'+

    # 2) "NOM / DENOM" syntax
    DENOM_START          # '/'       ->  (DENOM_SIGN, SMALL_DENOM)
    DENOM_SIGN           # [+-]      ->  (SMALL_DENOM)
    SMALL_DENOM          # [0-9]+    ->  (SMALL_DENOM, SMALL_DENOM_US, DENOM, DENOM_SPACE)
    SMALL_DENOM_US       # '_'       ->  (SMALL_DENOM)
    DENOM                # [0-9]+    ->  (DENOM, DENOM_US, DENOM_SPACE)
    DENOM_US             # '_'       ->  (DENOM)
    DENOM_SPACE          # '\s'+


cdef _raise_invalid_input(s):
    s = repr(s)
    if s[:2] in ('b"', "b'"):
        s = s[1:]
    raise ValueError(f'Invalid literal for Fraction: {s}') from None


cdef _raise_parse_overflow(s):
    s = repr(s)
    if s[0] == 'b':
        s = s[1:]
    raise ValueError(f"Exponent too large for Fraction: {s}") from None


cdef extern from *:
    """
    static CYTHON_INLINE int __QUICKTIONS_unpack_ustring(
            PyObject* string, Py_ssize_t *length, void** data, int *kind) {
        if (PyUnicode_READY(string) < 0) return -1;
        *kind   = PyUnicode_KIND(string);
        *length = PyUnicode_GET_LENGTH(string);
        *data   = PyUnicode_DATA(string);
        return 0;
    }
    #define __QUICKTIONS_char_at(data, kind, index) \
        (((kind == 1) ? (Py_UCS4) ((char*) data)[index] : (Py_UCS4) PyUnicode_READ(kind, data, index)))
    """
    int _unpack_ustring "__QUICKTIONS_unpack_ustring" (
        object string, Py_ssize_t *length, void **data, int *kind) except -1
    Py_UCS4 _char_at "__QUICKTIONS_char_at" (void *data, int kind, Py_ssize_t index)
    Py_UCS4 PyUnicode_READ(int kind, void *data, Py_ssize_t index)


cdef inline int _parse_digit(char** c_digits, Py_UCS4 c, int allow_unicode):
    cdef unsigned int unum
    cdef int num
    unum = (<unsigned int> c) - <unsigned int> '0'  # Relies on integer underflow for dots etc.
    if unum > 9:
        if not allow_unicode:
            return -1
        num = Py_UNICODE_TODECIMAL(c)
        if num == -1:
            return -1
        unum = <unsigned int> num
        c = <Py_UCS4> (num + c'0')
    if c_digits:
        c_digits[0][0] = <char> c
        c_digits[0] += 1
    return <int> unum


cdef inline object _parse_pylong(char* c_digits_start, char** c_digits_end):
    c_digits_end[0][0] = 0
    py_number = PyLong_FromString(c_digits_start, NULL, 10)
    c_digits_end[0] = c_digits_start  # reset
    return py_number


@cython.cdivision(True)
cdef tuple _parse_fraction(AnyString s, Py_ssize_t s_len, orig_str):
    """
    Parse a string into a number tuple: (numerator, denominator, is_normalised)
    """
    cdef Py_ssize_t pos, decimal_len = 0
    cdef Py_UCS4 c
    cdef ParserState state = BEGIN_SPACE

    cdef bint is_neg = False, exp_is_neg = False
    cdef int digit
    cdef unsigned int udigit
    cdef long long inum = 0, idecimal = 0, idenom = 0, iexp = 0
    cdef ullong igcd
    cdef object num = None, decimal, denom
    # 2^n > 10^(n * 5/17)
    cdef Py_ssize_t max_decimal_len = <Py_ssize_t> (sizeof(inum) * 8) * 5 // 17

    # Incremental Unicode iteration isn't in Cython yet.
    cdef int allow_unicode = AnyString is unicode
    cdef int s_kind = 1
    cdef void* s_data = NULL
    cdef char* cdata = NULL

    if AnyString is unicode:
        _unpack_ustring(s, &s_len, &s_data, &s_kind)
        if s_kind == 1:
            return _parse_fraction(<char*> s_data, s_len, orig_str)
        cdata = <char*> s_data
        cdata += 0  # mark used
    else:
        cdata = s

    # We collect the digits in inum / idenum as long as the value fits their integer size
    # and additionally in a char* buffer in case it grows too large.
    cdef bytes digits = b'\0' * s_len
    cdef char* c_digits_start = digits
    cdef char* c_digits = c_digits_start

    pos = 0
    while pos < s_len:
        c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
        pos += 1
        digit = _parse_digit(&c_digits, c, allow_unicode)
        if digit == -1:
            if c == u'/':
                if state == SMALL_NUM:
                    num = inum
                elif state in (NUM, NUM_SPACE):
                    num = _parse_pylong(c_digits_start, &c_digits)
                else:
                    _raise_invalid_input(orig_str)
                state = DENOM_START
                break
            elif c == u'.':
                if state in (BEGIN_SPACE, BEGIN_SIGN):
                    state = START_DECIMAL_DOT
                elif state == SMALL_NUM:
                    state = SMALL_DECIMAL_DOT
                elif state == NUM:
                    state = DECIMAL_DOT
                else:
                    _raise_invalid_input(orig_str)
                break
            elif c in u'eE':
                if state == SMALL_NUM:
                    num = inum
                elif state == NUM:
                    num = _parse_pylong(c_digits_start, &c_digits)
                else:
                    _raise_invalid_input(orig_str)
                state = EXP_E
                break
            elif c in u'-+':
                if state == BEGIN_SPACE:
                    is_neg = c == u'-'
                    state = BEGIN_SIGN
                else:
                    _raise_invalid_input(orig_str)
                continue
            elif c == u'_':
                if state == SMALL_NUM:
                    state = SMALL_NUM_US
                elif state == NUM:
                    state = NUM_US
                else:
                    _raise_invalid_input(orig_str)
                continue
            else:
                if c.isspace():
                    while pos < s_len:
                        c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                        if not c.isspace():
                            break
                        pos += 1

                    if state in (BEGIN_SPACE, NUM_SPACE):
                        continue
                    elif state == SMALL_NUM:
                        num = inum
                        state = NUM_SPACE
                    elif state == NUM:
                        num = _parse_pylong(c_digits_start, &c_digits)
                        state = NUM_SPACE
                    else:
                        _raise_invalid_input(orig_str)
                    continue

                _raise_invalid_input(orig_str)
                continue

        # normal digit found
        if state in (BEGIN_SPACE, BEGIN_SIGN, SMALL_NUM, SMALL_NUM_US):
            inum = inum * 10 + digit
            state = SMALL_NUM

            # fast-path for consecutive digits
            while pos < s_len and inum <= MAX_SMALL_NUMBER:
                c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                digit = _parse_digit(&c_digits, c, allow_unicode)
                if digit == -1:
                    break
                inum = inum * 10 + digit
                pos += 1

            if inum > MAX_SMALL_NUMBER:
                state = NUM
        elif state == NUM_US:
            state = NUM

        # We might have switched to NUM above, so continue right here in that case.
        if state == SMALL_NUM:
            pass  # handled above
        elif state == NUM:
            # fast-path for consecutive digits
            while pos < s_len:
                c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                digit = _parse_digit(&c_digits, c, allow_unicode)
                if digit == -1:
                    break
                pos += 1
        else:
            _raise_invalid_input(orig_str)

    if state == DENOM_START:
        # NUM '/'  |  SMALL_NUM '/'
        while pos < s_len:
            c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
            if not c.isspace():
                break
            pos += 1

        # Start fresh digits collection.
        c_digits = c_digits_start
        while pos < s_len:
            c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
            pos += 1
            digit = _parse_digit(&c_digits, c, allow_unicode)
            if digit == -1:
                if c in u'-+':
                    if state == DENOM_START:
                        is_neg ^= (c == u'-')
                        state = DENOM_SIGN
                    else:
                        _raise_invalid_input(orig_str)
                    continue
                elif c == u'_':
                    if state == SMALL_DENOM:
                        state = SMALL_DENOM_US
                    elif state == DENOM:
                        state = DENOM_US
                    else:
                        _raise_invalid_input(orig_str)
                    continue
                else:
                    if c.isspace():
                        while pos < s_len:
                            c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                            if not c.isspace():
                                break
                            pos += 1

                        if state in (DENOM_START, DENOM_SPACE):
                            pass
                        elif state == SMALL_DENOM:
                            denom = idenom
                        elif state == DENOM:
                            denom = _parse_pylong(c_digits_start, &c_digits)
                        else:
                            _raise_invalid_input(orig_str)
                        state = DENOM_SPACE
                        continue

                    _raise_invalid_input(orig_str)
                    continue

            # normal digit found
            if state in (DENOM_START, DENOM_SIGN, SMALL_DENOM, SMALL_DENOM_US):
                idenom = idenom * 10 + digit
                state = SMALL_DENOM

                # fast-path for consecutive digits
                while pos < s_len and idenom <= MAX_SMALL_NUMBER:
                    c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                    digit = _parse_digit(&c_digits, c, allow_unicode)
                    if digit == -1:
                        break
                    idenom = idenom * 10 + digit
                    pos += 1

                if idenom > MAX_SMALL_NUMBER:
                    state = DENOM
            elif state == DENOM_US:
                state = DENOM

            # We might have switched to DENOM above, so continue right here in that case.
            if state == SMALL_DENOM:
                pass  # handled above
            elif state == DENOM:
                # fast-path for consecutive digits
                while pos < s_len:
                    c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                    digit = _parse_digit(&c_digits, c, allow_unicode)
                    if digit == -1:
                        break
                    pos += 1
            else:
                _raise_invalid_input(orig_str)

    elif state in (SMALL_DECIMAL_DOT, START_DECIMAL_DOT):
        # SMALL_NUM '.'  | '.'
        while pos < s_len:
            c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
            pos += 1
            digit = _parse_digit(&c_digits, c, allow_unicode)
            if digit == -1:
                if c == u'_':
                    if state == SMALL_DECIMAL:
                        state = SMALL_DECIMAL_US
                    else:
                        _raise_invalid_input(orig_str)
                    continue
                elif c in u'eE':
                    if state in (SMALL_DECIMAL_DOT, SMALL_DECIMAL):
                        num = inum
                    else:
                        _raise_invalid_input(orig_str)
                    state = EXP_E
                    break
                else:
                    if c.isspace():
                        while pos < s_len:
                            c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                            if not c.isspace():
                                break
                            pos += 1

                        if state in (SMALL_DECIMAL, SMALL_DECIMAL_DOT):
                            num = inum
                            state = SMALL_END_SPACE
                        else:
                            _raise_invalid_input(orig_str)
                        continue

                    _raise_invalid_input(orig_str)
                    continue

            # normal digit found
            if state in (START_DECIMAL_DOT, SMALL_DECIMAL_DOT, SMALL_DECIMAL, SMALL_DECIMAL_US):
                inum = inum * 10 + digit
                decimal_len += 1
                state = SMALL_DECIMAL

                # fast-path for consecutive digits
                while pos < s_len and inum <= MAX_SMALL_NUMBER and decimal_len < max_decimal_len:
                    c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                    digit = _parse_digit(&c_digits, c, allow_unicode)
                    if digit == -1:
                        break
                    inum = inum * 10 + digit
                    decimal_len += 1
                    pos += 1

                if inum > MAX_SMALL_NUMBER or decimal_len >= max_decimal_len:
                    state = DECIMAL
                    break
            else:
                _raise_invalid_input(orig_str)

    if state in (DECIMAL_DOT, DECIMAL):
        # NUM '.'  |  SMALL_DECIMAL->DECIMAL
        while pos < s_len:
            c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
            pos += 1
            digit = _parse_digit(&c_digits, c, allow_unicode)
            if digit == -1:
                if c == u'_':
                    if state == DECIMAL:
                        state = DECIMAL_US
                    else:
                        _raise_invalid_input(orig_str)
                    continue
                elif c in u'eE':
                    if state in (DECIMAL_DOT, DECIMAL):
                        num = _parse_pylong(c_digits_start, &c_digits)
                    else:
                        _raise_invalid_input(orig_str)
                    state = EXP_E
                    break
                else:
                    if c.isspace():
                        if state in (DECIMAL, DECIMAL_DOT):
                            state = END_SPACE
                        else:
                            _raise_invalid_input(orig_str)
                        break

                    _raise_invalid_input(orig_str)
                    continue

            # normal digit found
            if state in (DECIMAL_DOT, DECIMAL, DECIMAL_US):
                decimal_len += 1
                state = DECIMAL

                # fast-path for consecutive digits
                while pos < s_len:
                    c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                    digit = _parse_digit(&c_digits, c, allow_unicode)
                    if digit == -1:
                        break
                    decimal_len += 1
                    pos += 1
            else:
                _raise_invalid_input(orig_str)

    if state == EXP_E:
        # (SMALL_) NUM ['.' DECIMAL] 'E'
        while pos < s_len:
            c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
            pos += 1
            digit = _parse_digit(NULL, c, allow_unicode)
            if digit == -1:
                if c in u'-+':
                    if state == EXP_E:
                        exp_is_neg = c == u'-'
                        state = EXP_SIGN
                    else:
                        _raise_invalid_input(orig_str)
                    continue
                elif c == u'_':
                    if state == EXP:
                        state = EXP_US
                    else:
                        _raise_invalid_input(orig_str)
                    continue
                else:
                    if c.isspace():
                        if state == EXP:
                            state = END_SPACE
                        else:
                            _raise_invalid_input(orig_str)
                        break

                    _raise_invalid_input(orig_str)
                    continue

            # normal digit found
            if state in (EXP_E, EXP_SIGN, EXP, EXP_US):
                iexp = iexp * 10 + digit
                state = EXP

                # fast-path for consecutive digits
                while pos < s_len and iexp <= MAX_SMALL_NUMBER:
                    c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
                    digit = _parse_digit(NULL, c, allow_unicode)
                    if digit == -1:
                        break
                    iexp = iexp * 10 + digit
                    pos += 1

                if iexp > MAX_SMALL_NUMBER:
                    _raise_parse_overflow(orig_str)
            else:
                _raise_invalid_input(orig_str)

    if state in (END_SPACE, SMALL_END_SPACE, DENOM_SPACE):
        while pos < s_len:
            c = _char_at(s_data, s_kind, pos) if AnyString is unicode else cdata[pos]
            if not c.isspace():
                break
            pos += 1

    if pos < s_len :
        _raise_invalid_input(orig_str)

    cdef bint is_normalised = False
    if state in (SMALL_NUM, SMALL_DECIMAL, SMALL_DECIMAL_DOT, SMALL_END_SPACE):
        # Special case for 'small' numbers: normalise directly in C space.
        if inum and decimal_len:
            # Only need to normalise if the numerator contains factors of a power of 10 (2 or 5).
            if inum & 1 == 0 or inum % 5 == 0:
                idenom = _c_pow10(decimal_len)
                igcd = _c_gcd(inum, idenom)
                if igcd > 1:
                    inum //= igcd
                    denom = idenom // igcd
                else:
                    denom = pow10(decimal_len)
            else:
                denom = pow10(decimal_len)
        else:
            denom = 1
        if is_neg:
            inum = -inum
        return inum, denom, True

    elif state == SMALL_DENOM:
        denom = idenom
    elif state in (NUM, DECIMAL, DECIMAL_DOT):
        is_normalised = True  # will be repaired below for iexp < 0
        denom = 1
        num = _parse_pylong(c_digits_start, &c_digits)
    elif state == DENOM:
        denom = _parse_pylong(c_digits_start, &c_digits)
    elif state in (NUM_SPACE, EXP, END_SPACE):
        is_normalised = True
        denom = 1
    elif state == DENOM_SPACE:
        pass
    else:
        _raise_invalid_input(orig_str)

    if decimal_len > MAX_SMALL_NUMBER:
        _raise_parse_overflow(orig_str)
    if exp_is_neg:
        iexp = -iexp
    iexp -= decimal_len

    if iexp > 0:
        num *= pow10(iexp)
    elif iexp < 0:
        # Only need to normalise if the numerator contains factors of a power of 10 (2 or 5).
        is_normalised = num & 1 != 0 and num % 5 != 0
        denom = pow10(-iexp)

    if is_neg:
        num = -num

    return num, denom, is_normalised
