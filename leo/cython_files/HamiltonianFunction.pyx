import numpy as np
cimport numpy as np
np.import_array()
from scipy.sparse import csr_matrix

cdef class HamiltonianSubroutine:
    cpdef void apply(self, np.ndarray[np.float64_t, ndim=1] state, np.ndarray[np.float64_t, ndim=1] output) except *:
        return 
    
    