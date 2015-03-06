cdef extern from *:
    ctypedef long Py_hash_t

cdef class Fraction:
    cdef _numerator
    cdef _denominator
    cdef Py_hash_t _hash
