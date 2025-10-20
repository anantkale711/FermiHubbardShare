#from __future__ import print_function
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np
np.import_array()

from numpy cimport uint32_t, int8_t

cdef double double_tol = 2e-16 

cdef long find_state(long s, s_list):
    ''' finds the location of s in s_list using binary search.
    Returns b such that s_list[b] = s '''
    cdef Py_ssize_t bm = 0, b=0
    cdef Py_ssize_t bM = len(s_list)-1
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

def create_fermionic_hopping_operator_ij(int i, int j, states):
    # a_i^\dagger a_j (for spin sigma = 0 or 1)
    cdef int D, c=0
    
    D = len(states)
    
    row = np.zeros(D, dtype = np.uint32)
    col = np.zeros(D, dtype = np.uint32)
    data = np.zeros(D, dtype = np.int8)

    cdef uint32_t[:] row_view = row
    cdef uint32_t[:] col_view = col
    cdef int8_t[:] data_view = data
    

#     c = 0     # counter
    
#     cdef int si, ni, sj, nj, low_s, high_s, spins_in_between, idx1
    cdef long state, new_state, idx, idx2, fermions_in_between
    cdef int low, high, ni, nj, phase
    if i == j:
        for idx in range(D):
            state = states[idx]
            #Op[idx, idx] += ni
            row_view[c] = idx
            col_view[c] = idx
            data_view[c] = get_bit_j_fast(state, j)
            c += 1
    else:
        low, high = (i,j) if (i<=j) else (j,i) 
        mask = (1<<(high+1) - 1) - (1<<(low+1) - 1)
        for idx in range(D):
            state = states[idx]
            # assume i<=j
            # if i > j, then use transpose, dont recalculate
            
            ni = get_bit_j_fast(state, low)
            nj = get_bit_j_fast(state, high)
            if (ni == 0 and nj == 1):
                fermions_in_between = count_bits(state & mask)
                phase = 1 - 2 * (fermions_in_between % 2)
                new_state = flip_bit2_fast(state, i, j)
                idx2 = find_state(new_state, states)
                
                row_view[c] = idx2
                col_view[c] = idx
                data_view[c] = phase
                c += 1
            # else:# do nothing
    # print(c)
    Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.int8)
    Opcsr = Op.tocsr()
    return Opcsr 

def find_state_long(s, s_list):
    ''' finds the location of state in tag_list using binary search.
    tag_list must be sorted.
    Returns idx such that tag_list[idx] = tag '''
    cdef unsigned long long bm = 0
    cdef unsigned long long bM = np.shape(s_list)[0]-1
    cdef unsigned long long b = 0
    while(bm <= bM):
        b = (bM + bm)//2
        if s < s_list[b]:
            bM = b-1
        elif s > s_list[b]:
            bm = b+1
        else:
            return b
    raise NameError('Couldnt find state with tag %f'%s)
    return None


def find_state_with_tag(tag, tag_list):
    ''' finds the location of state in tag_list using binary search.
    tag_list must be sorted.
    Returns idx such that tag_list[idx] = tag '''
    cdef int bm = 0
    cdef int bM = np.shape(tag_list)[0]-1
    cdef int b = 0
    while(bm <= bM):
        b = int((bM + bm)/2)
        if tag < (tag_list[b] - double_tol):
            bM = b-1
        elif tag > (tag_list[b] + double_tol):
            bm = b+1
        else:
            return b
    #print('State with tag %f not found! Tolerance: %f'%(tag, double_tol))
    raise NameError('Couldnt find state with tag %f'%tag)
    return None

def create_fermionic_hopping_operator_ij_sigma(i, j, sigma, vecs_sorted, tags_sorted):
    # a_i^\dagger a_j (for spin sigma = 0 or 1)
    cdef int D, N_sites, c=0
    
    D, N_sites = np.shape(vecs_sorted)
    row = np.zeros(D, dtype = np.uint32)
    col = np.zeros(D, dtype = np.uint32)
    data = np.zeros(D, dtype = np.int8)
    
    sites = np.arange(1, N_sites + 1, 1)
    temp = np.power(4, N_sites - sites, dtype = np.float64)
    
    vec1 = np.zeros(N_sites, dtype = np.int8)
    lookup_n = np.array([[0, 1, 0, 1],
                         [0, 0, 1, 1]], dtype = np.int8)
    c = 0     # counter
    
    cdef int si, ni, sj, nj, low_s, high_s, spins_in_between, idx1
    cdef double tag
    if i == j:
        for idx in range(D):
            si = vecs_sorted[idx, i]
            ni = lookup_n[sigma, si]
            #Op[idx, idx] += vecs_sorted[idx, i]
            row[c] = idx
            col[c] = idx
            data[c] = ni
            c += 1
    else:
        for idx in range(D):
            si = vecs_sorted[idx, i]
            sj = vecs_sorted[idx, j]
            ni = lookup_n[sigma, si]
            nj = lookup_n[sigma, sj]

            if (ni == 0 and nj == 1):
                # do something
                low_s = min(i,j)
                high_s = max(i,j)
                spins_in_between = np.sum(lookup_n[sigma, vecs_sorted[idx,low_s+1:high_s]])
                phase = int((-1)**spins_in_between)
                vec1[:] = vecs_sorted[idx]
                vec1[i] = vecs_sorted[idx, i] + (sigma + 1) 
                vec1[j] = vecs_sorted[idx, j] - (sigma + 1)
                #tag = tag_state_FH(vec1, N_sites)
                tag = np.sum(temp * (vec1).astype(np.float64))
                try:
                    idx1 = find_state_with_tag(tag, tags_sorted)
                except NameError:
#                     print('idx: %d'%idx)
#                     print('vec: ', vecs_sorted[idx])
#                     print('vec1: ', vec1)
                    raise
                #Op[idx1, idx] += phase  ( = +-1)
                row[c] = idx1
                col[c] = idx
                data[c] = phase
                c += 1
            # else:# do nothing
    # print(c)
    Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.int8)
    Opcsr = Op.tocsr()
    return Opcsr    
