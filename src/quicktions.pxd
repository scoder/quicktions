# cython: language_level=3str
## cython: profile=True

cpdef _gcd(a, b)

cdef class Fraction:
    cdef _numerator
    cdef _denominator
    cdef Py_hash_t _hash

    cpdef limit_denominator(self, max_denominator=*)
    cpdef conjugate(self)
    cdef _eq(a, b)
    cdef _richcmp(self, other, int op)
