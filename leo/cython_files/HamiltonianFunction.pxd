import numpy as np
cimport numpy as np
np.import_array()
cdef class HamiltonianSubroutine:
    cpdef void apply(self, np.ndarray[np.float64_t, ndim=1] state, np.ndarray[np.float64_t, ndim=1] output) except *