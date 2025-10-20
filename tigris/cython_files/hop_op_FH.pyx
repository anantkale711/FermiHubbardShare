from __future__ import print_function
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

import numpy as np
cimport numpy as np
np.import_array()

cdef double double_tol = 2e-16 

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
    print('State with tag %f not found! Tolerance: %f'%(tag, double_tol))
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
                    print('idx: %d'%idx)
                    print('vec: ', vecs_sorted[idx])
                    print('vec1: ', vec1)
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





# def create_fermionic_operator_mixed(i, j, sigma_i, sigma_j, vecs_sorted, tags_sorted):
#     # a_{i,si}^\dagger a_{j,sj} (for spin sigma = 0 or 1)
#     cdef int D, N_sites, c=0
    
#     D, N_sites = np.shape(vecs_sorted)
#     row = np.zeros(D, dtype = np.uint32)
#     col = np.zeros(D, dtype = np.uint32)
#     data = np.zeros(D)
    
#     sites = np.arange(1, N_sites + 1, 1)
#     temp = np.power(4, N_sites - sites, dtype = np.uint64)
    
#     vec1 = np.zeros(N_sites, dtype = np.int8)
#     lookup_n = np.array([[0, 1, 0, 1],
#                          [0, 0, 1, 1]], dtype = np.int8)
#     c = 0     # counter
    
#     cdef int si, ni, sj, nj, tag, idx1, spins_to_left_i, spins_to_left_j
#     if i == j and sigma_i == sigma_j:
#         sigma = sigma_j
#         for idx in range(D):
#             si = vecs_sorted[idx, i]
#             ni = lookup_n[sigma, si]
#             #Op[idx, idx] += vecs_sorted[idx, i]
#             row[c] = idx
#             col[c] = idx
#             data[c] = ni
#             c += 1
            
#     else:
#         for idx in range(D):
#             si = vecs_sorted[idx, i]
#             sj = vecs_sorted[idx, j]
#             ni = lookup_n[sigma_i, si]
#             nj = lookup_n[sigma_j, sj]

#             if (ni == 0 and nj == 1):  # do something
#                 # spin down are to the right of all spin up operators
#                 vec1[:] = vecs_sorted[idx]
                
#                 # count fermions to the left of j
#                 # count all fermions of spin sigma_j + all up-spin fermions if sigma_j is a down spin
#                 spins_to_left_j = np.sum(lookup_n[sigma_j, vec1[:j]]) + sigma_j * np.sum(lookup_n[0, vec1[:]])
#                 # annihilate fermion at j
#                 vec1[j] = vec1[j] - (sigma_j + 1)
                
#                 # count spins to the left of i (same as above)
#                 spins_to_left_i = np.sum(lookup_n[sigma_i, vec1[:i]]) + sigma_j * np.sum(lookup_n[0, vec1[:]])
#                 # create fermion at site i
#                 vec1[i] = vec1[i] + (sigma_i + 1) 
                
#                 # phase comes from both operators
#                 phase = (-1)**spins_to_left_i * (-1)**spins_to_left_j
#                 #tag = tag_state_FH(vec1, N_sites)
#                 tag = np.sum(temp * vec1)
#                 idx1 = hams.find_state_with_tag(int(tag), tags_sorted)

#                 #Op[idx1, idx] += phase  ( = +-1)
#                 row[c] = idx1
#                 col[c] = idx
#                 data[c] = phase
#                 c += 1
        
#     # print(c)
#     Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.float64)
#     Opcsr = Op.tocsr()
#     return Opcsr    

