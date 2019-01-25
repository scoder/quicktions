# cython: language_level=3str
## cython: profile=True

cdef extern from *:
    """
    #if PY_VERSION_HEX < 0x030500F0 || !CYTHON_COMPILING_IN_CPYTHON
        #define _PyLong_GCD(a, b) (NULL)
    #endif
    """
    # CPython 3.5+ has a fast PyLong GCD implementation that we can use.
    int PY_VERSION_HEX
    int IS_CPYTHON "CYTHON_COMPILING_IN_CPYTHON"
    _PyLong_GCD(a, b)

ctypedef unsigned long long ullong
ctypedef unsigned long ulong
ctypedef unsigned int uint

ctypedef fused cunumber:
    ullong
    ulong
    uint

cpdef _gcd(a, b)
cdef ullong _abs(long long x)
cdef cunumber _igcd(cunumber a, cunumber b)
cdef cunumber _ibgcd(cunumber a, cunumber b)
cdef _py_gcd(ullong a, ullong b)
cdef _gcd_fallback(a, b)

cdef class Fraction:
    cdef _numerator
    cdef _denominator
    cdef Py_hash_t _hash

    cpdef limit_denominator(self, max_denominator=*)
    cpdef conjugate(self)
    cdef _eq(a, b)
    cdef _richcmp(self, other, int op)
