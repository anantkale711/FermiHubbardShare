import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

np.import_array()



cdef long get_bit_j_fast(long n, int j):
    cdef long temp = n>>j
    cdef long mask = 1
    cdef long result = temp & mask
    return result

cdef long flip_bit2_fast(long n, int i, int j):
    cdef long mask = (1<<i) ^ (1<<j)
    cdef long result = n ^ mask
    return result

cdef long count_bits(long n):
    cdef long count = 0
    while (n > 0):
        count += n & 1
        n >>= 1
    return count


cdef long find_state(long s, s_list):
    ''' finds the location of s in s_list using binary search.
    Returns b such that s_list[b] = s '''
    cdef int bm = 0
    cdef int bM = len(s_list)-1
    while(bm <= bM):
        b = int((bM + bm)/2)
        if s < s_list[b]:
            bM = b-1
        elif s > s_list[b]:
            bm = b+1
        else:
            return b
    #print('State %d not found!'%s)
    raise NameError('Couldnt find state with tag %d'%s) 
    return None

def generate_s_sector_states(s, L):
    # s = #ups - #downs
    cdef dimH = 1<<L
    states = np.zeros(dimH, dtype=np.uint32)
    cdef long count=0, a=0, n_1s=0, n_0s
    cdef long N_sites=L
    cdef long Sz = s
    while a < dimH:
        n_1s = count_bits(a)
        n_0s = N_sites - n_1s
        # zero is up, one is down
        if n_0s - n_1s == Sz:
            states[count]=a
            count += 1
        a += 1
    return states[:count]


def create_H_Heisenberg_s_sector(connectivity_list, states):
    cdef long L = len(connectivity_list)
    cdef long D = len(states)
    row = np.zeros(10*L*D, dtype = np.uint32)
    col = np.zeros(10*L*D, dtype = np.uint32)
    data = np.zeros(10*L*D, dtype = np.int8)
    cdef long count = 0, idx = 0, idx2 = 0
    cdef long a = 0, j=0, ak=0, aj=0, k=0
    cdef clist = np.array(connectivity_list, dtype=long)
    while idx < D:
        a = states[idx]
        # sigma_+ sigma_- term and sigma_z sigma_z term
        j=0
        while j < L:
            aj = get_bit_j_fast(a, j)
            for k in clist[j]:
                ak = get_bit_j_fast(a, k)
                if aj == ak:
                    # H[idx, idx] += 1 / 4
                    row[count]=  idx
                    col[count] = idx
                    data[count] = 1
                    count += 1
                if aj != ak:
                    # H[idx, idx] += - 1 / 4
                    row[count] = idx
                    col[count] = idx
                    data[count] = -1
                    count += 1

                    b = flip_bit2_fast(a, j, k)
                    idx2 = find_state(b, states)
                    # H[idx2, idx] = 1 / 2
                    row[count] = idx2
                    col[count] = idx
                    data[count] = 2
                    count += 1
            j += 1
        idx += 1
    H = coo_matrix((data[:count+1], (row[:count+1], col[:count+1])), shape = (D, D), dtype = np.int8)
    Hcsr = H.tocsr()
    return Hcsr