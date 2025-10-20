from __future__ import print_function
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

import numpy as np
cimport numpy as np

np.import_array()

from numpy cimport uint32_t

cdef uint32_t find_state(uint32_t s, np.ndarray[uint32_t, ndim=1] s_list):
    ''' finds the location of s in s_list using binary search.
    Returns b such that s_list[b] = s '''
    cdef uint32_t bm = 0, b=0
    cdef uint32_t bM = len(s_list)-1
    while(bm <= bM):
#         b = int((bM + bm)/2)
        b = (bM + bm)//2

        if s < s_list[b]:
            bM = b-1
        elif s > s_list[b]:
            bm = b+1
        else:
            return b
    #print('State %d not found!'%s)
    raise NameError('Couldnt find state with tag %d'%s) 
    return None

cdef uint32_t get_bit_j_fast(uint32_t n, uint32_t j):
    cdef uint32_t temp = n>>j
    cdef uint32_t mask = 1
    cdef uint32_t result = temp & mask
    return result

cdef uint32_t flip_bit2_fast(uint32_t n, uint32_t i, uint32_t j):
    cdef uint32_t mask = (1<<i) ^ (1<<j)
    cdef uint32_t result = n ^ mask
    return result

cdef uint32_t count_1s(uint32_t n):
    cdef uint32_t count = 0
    while (n > 0):
        count += n & 1
        n >>= 1
    return count

def test(int i1, int j1, int sigma, int N_sites, np.ndarray[uint32_t, ndim=1] states):
    cdef uint32_t idx=0, idx2=0, mask, state2, state
    D = len(states)
    i = i1 + (1-sigma)*N_sites
    j = j1 + (1-sigma)*N_sites
    
    ibar = i1 + (sigma)*N_sites
    jbar = j1 + (sigma)*N_sites
    
    for idx in range(D):
        state = states[idx]
        ni = get_bit_j_fast(state, i)
        nj = get_bit_j_fast(state, j)
        nibar = get_bit_j_fast(state, ibar)
        njbar = get_bit_j_fast(state, jbar)
        if (ni == 0 and nj == 1) and (nibar == 0 and njbar == 0):
            state2 = flip_bit2_fast(state, i, j)
            if state2 > 3221237760:
                print(idx, state, state2)
#             idx2 = find_state(state2, states)
#             if state2 < 0:
#                 print(idx, state, state2)
    return

def test2(np.ndarray[uint32_t, ndim=1] states):
    cdef uint32_t idx=0, idx2=0, mask, state2, state
    D = len(states)
    for idx in range(D):
        state = states[idx]
        idx2 = find_state(state, states)
        if idx != idx2:
            print(idx, idx2, state)
    return

# def create_hole_hopping_operator_ij_tJmodel(int i, int j, int sigma, int N_sites, np.ndarray[uint32_t, ndim=1] states):
#     # a_i^\dagger a_j (for spin sigma = 0 or 1)
#     cdef uint32_t D, c=0
#     cdef uint32_t ni, nj, nibar, njbar, phase, fermions_in_between, low, high
#     cdef uint32_t idx=0, idx2=0, mask, state2, state
    
#     D = len(states)
#     row = np.zeros(D, dtype = np.uint64)
#     col = np.zeros(D, dtype = np.uint64)
#     data = np.zeros(D, dtype = np.float64)
    
#     c = 0     # counter
#     i = i + (1-sigma)*N_sites
#     j = j + (1-sigma)*N_sites
    
    
#     if i == j:
#         for idx in range(D):
#             state = states[idx]
#             #Op[idx, idx] += ni
#             row[c] = idx
#             col[c] = idx
#             data[c] = get_bit_j_fast(state, j)
#             c += 1
#     else:
# #         low, high = (i,j) if (i<=j) else (j,i) 
# #         mask = (1<<(high+1) - 1) - (1<<(low+1) - 1)
#         mask = (1<<(j+1) - 1) - (1<<(i+1) - 1)
#         for idx in range(D):
#             state = states[idx]
#             # assume i<=j
#             # if i > j, then use transpose, dont recalculate
            
#             ni = get_bit_j_fast(state, i)
#             nj = get_bit_j_fast(state, j)
            
            
#             nibar = get_bit_j_fast(state, i+(2*sigma-1)*N_sites)
#             njbar = get_bit_j_fast(state, j+(2*sigma-1)*N_sites)
            
#             if (ni == 0 and nj == 1) and (nibar == 0 and njbar == 0):
#                 fermions_in_between = count_1s(state & mask)
                
#                 phase = 1 - 2 * (fermions_in_between % 2)
#                 state2 = flip_bit2_fast(state, i, j)
#                 idx2 = find_state(state2, states)
                
#                 row[c] = idx2
#                 col[c] = idx
#                 data[c] = phase
#                 c += 1
# #             else:# do nothing
# #                 print("rejected: ", ni, nj, nibar, njbar)
    
#     # print(c)
#     Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.float64)
#     Opcsr = Op.tocsr()
#     return Opcsr 

def create_hole_hopping_operator_ij_tJmodel(int i1, int j1, int sigma, int N_sites, np.ndarray[uint32_t, ndim=1] states):
    # a_i^\dagger a_j (for spin sigma = 0 or 1)
    cdef uint32_t D, c=0
    cdef uint32_t ni, nj, nibar, njbar, 
    cdef int phase, phasebar, 
    cdef uint32_t fermions_in_between, fermions_in_between_bar, low, high
    cdef uint32_t idx=0, idx2=0, mask, maskbar, state, state2_temp, state2
    cdef uint32_t i, j, ibar, jbar
    
    D = len(states)
    row = np.zeros(D, dtype = np.uint64)
    col = np.zeros(D, dtype = np.uint64)
    data = np.zeros(D, dtype = np.float64)
    
    c = 0     # counter
    i = i1 + (1-sigma)*N_sites
    j = j1 + (1-sigma)*N_sites
    
    ibar = i1 + (sigma)*N_sites
    jbar = j1 + (sigma)*N_sites
    
    if i == j:
        for idx in range(D):
            state = states[idx]
            #Op[idx, idx] += ni
            row[c] = idx
            col[c] = idx
            data[c] = get_bit_j_fast(state, j)
            c += 1
    else:
#         low, high = (i,j) if (i<=j) else (j,i) 
#         mask = (1<<(high+1) - 1) - (1<<(low+1) - 1)
        mask = (1<<(j+1) - 1) - (1<<(i+1) - 1)
        for idx in range(D):
            state = states[idx]
            # assume i<=j
            # if i > j, then use transpose, dont recalculate
            
            ni    = get_bit_j_fast(state, i)
            nj    = get_bit_j_fast(state, j)
            nibar = get_bit_j_fast(state, ibar)
            njbar = get_bit_j_fast(state, jbar)
            
            if (ni == 0 and nj == 1) and (nibar == 0 and njbar == 0):
                fermions_in_between = count_1s(state & mask)
                
                phase = 1 - 2 * (fermions_in_between % 2)
                state2 = flip_bit2_fast(state, i, j)
                idx2 = find_state(state2, states)
                
                row[c] = idx2
                col[c] = idx
                data[c] = phase
                c += 1
#             else:# do nothing
#                 print("rejected: ", ni, nj, nibar, njbar)
    
    # print(c)
    Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.float64)
    Opcsr = Op.tocsr()
    return Opcsr 

def create_electron_exchange_operator_ij_tJmodel(int i, int j, int N_sites, np.ndarray[uint32_t, ndim=1] states):
    # c_i^\dagger_sigma c_j_sigma c_j^\dagger_sigmabar c_i_sigmabar
    # Assume i < j
    
    cdef uint32_t D, c=0
    cdef uint32_t ibar, jbar, ni, nj, nibar, njbar, 
    cdef int phase, phasebar, 
    cdef uint32_t fermions_in_between, fermions_in_between_bar, low, high
    cdef uint32_t idx=0, idx2=0, mask, maskbar, state, state2_temp, state2
    
    if i == j: return None
    
    D = len(states)
    
    row = np.zeros(4*D, dtype = np.uint64)
    col = np.zeros(4*D, dtype = np.uint64)
    data = np.zeros(4*D, dtype = np.float64)
    
    c = 0     # counter
    # i,j for spin down
    # ibar, jbar for spin up
    ibar = i + N_sites  
    jbar = j + N_sites

    mask = (1<<(j+1) - 1) - (1<<(i+1) - 1)
    maskbar = (1<<(jbar+1) - 1) - (1<<(ibar+1) - 1)
    
#     print(i,j, decimal_to_bin_array(mask, 2*N_sites))
#     print(ibar,jbar, decimal_to_bin_array(maskbar, 2*N_sites))
    for idx in range(D):
        state = states[idx]
        # assume i<=j
        # if i > j, then use transpose, dont recalculate

        ni = get_bit_j_fast(state, i)
        nj = get_bit_j_fast(state, j)
        nibar = get_bit_j_fast(state, ibar)
        njbar = get_bit_j_fast(state, jbar)
#             print(ni, nj, nibar, njbar)

        if ((ni == 0 and nj == 1) and (njbar == 0 and nibar == 1)) or \
           ((nibar == 0 and njbar == 1) and (nj == 0 and ni == 1)):
            
            fermions_in_between = count_1s(state & mask)
            fermions_in_between_bar = count_1s(state & maskbar) + 1 # cdag_jbar c_ibar and ibar < jbar
            phase = 1 - 2 * (fermions_in_between % 2)
            phasebar = 1 - 2 * (fermions_in_between_bar % 2)
            state2_temp = flip_bit2_fast(state, i, j)
            state2 = flip_bit2_fast(state2_temp, ibar, jbar)

            idx2 = find_state(state2, states)

#             print(state, decimal_to_bin_array(state, 2*N_sites), 
#                       state2, decimal_to_bin_array(state2, 2*N_sites), 
#                       idx, idx2, phase*phasebar)

            row[c] = idx2
            col[c] = idx
            data[c] = phase * phasebar
            c += 1
#             else:# do nothing
#                 print("rejected: ", ni, nj, nibar, njbar)
    # print(c)
    Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.float64)
    Opcsr = Op.tocsr()
    return Opcsr 

def create_3site_spin_preserve_hop_ilj_tJmodel(int i, int l, int j, int N_sites, np.ndarray[uint32_t, ndim=1] states):
    # sum_sigma  c_i^\dagger_sigmabar c_l_sigmabar c_l^\dagger_sigmabar c_j_sigma
#     cdef int D, N_sites, c=0
    if i == l or l==j or i==j: return None
    
    cdef uint32_t D, c=0
    cdef uint32_t ibar, lbar, jbar, ni, nl, nj, nibar, nlbar, njbar, 
    cdef int phase, phasebar
    cdef uint32_t mask_il, mask_ilbar, mask_lj, mask_ljbar
    cdef int phase_il, phase_ilbar, phase_lj, phase_ljbar
    cdef uint32_t fermions_in_between_il, fermions_in_between_ilbar, fermions_in_between_lj, fermions_in_between_ljbar,
    cdef uint32_t idx=0, idx2=0, state, state2_temp, state2
    
    
    D = len(states)
    
    row = np.zeros(4*D, dtype = np.uint64)
    col = np.zeros(4*D, dtype = np.uint64)
    data = np.zeros(4*D, dtype = np.float64)
    
    c = 0     # counter
    # i,j for spin down
    # ibar, jbar for spin up
    ibar = i + N_sites  
    lbar = l + N_sites
    jbar = j + N_sites
    
    
    mask_il = ((1<<(l+1) - 1) - (1<<(i+1) - 1)) if i < l else ((1<<(i+1) - 1) - (1<<(l+1) - 1))
    mask_ilbar = ((1<<(lbar+1) - 1) - (1<<(ibar+1) - 1)) if i < l else ((1<<(ibar+1) - 1) - (1<<(lbar+1) - 1))
    
    mask_lj = ((1<<(j+1) - 1) - (1<<(l+1) - 1)) if l < j else ((1<<(l+1) - 1) - (1<<(j+1) - 1))
    mask_ljbar = ((1<<(jbar+1) - 1) - (1<<(lbar+1) - 1)) if l < j else ((1<<(lbar+1) - 1) - (1<<(jbar+1) - 1))

#     print(i,l,j)
#     print(ibar, lbar, jbar)
#         , decimal_to_bin_array(mask, 2*N_sites))
    for idx in range(D):
        state = states[idx]
        
        ni = get_bit_j_fast(state, i)
        nl = get_bit_j_fast(state, l)
        nj = get_bit_j_fast(state, j)
        nibar = get_bit_j_fast(state, ibar)
        nlbar = get_bit_j_fast(state, lbar)
        njbar = get_bit_j_fast(state, jbar)
#             print(ni, nj, nibar, njbar)

        if ((nibar == 0 and nlbar == 1) and (nl == 0 and nj == 1) and (ni == 0)):
            fermions_in_between_ilbar = count_1s(state & mask_ilbar)
            fermions_in_between_ilbar += 1 if i > l else 0
            
            fermions_in_between_lj = count_1s(state & mask_lj)
            fermions_in_between_lj += 1 if l > j else 0

            phase_ilbar = 1 - 2 * (fermions_in_between_ilbar % 2)
            phase_lj = 1 - 2 * (fermions_in_between_lj % 2)
            state2_temp = flip_bit2_fast(state, ibar, lbar)
            state2 = flip_bit2_fast(state2_temp, l, j)

            idx2 = find_state(state2, states)

#             print(state, decimal_to_bin_array(state, 2*N_sites), 
#                   state2, decimal_to_bin_array(state2, 2*N_sites), 
#                   idx, idx2)

            row[c] = idx2
            col[c] = idx
            data[c] = phase_ilbar * phase_lj
            c += 1
        elif ((ni == 0 and nl == 1) and (nlbar == 0 and njbar == 1) and (nibar==0)):
            fermions_in_between_il = count_1s(state & mask_il)
            fermions_in_between_il += 1 if i > l else 0
            
            fermions_in_between_ljbar = count_1s(state & mask_ljbar)
            fermions_in_between_ljbar += 1 if l > j else 0

            phase_il = 1 - 2 * (fermions_in_between_il % 2)
            phase_ljbar = 1 - 2 * (fermions_in_between_ljbar % 2)
            state2_temp = flip_bit2_fast(state, i, l)
            state2 = flip_bit2_fast(state2_temp, lbar, jbar)

            idx2 = find_state(state2, states)

#             print(state, decimal_to_bin_array(state, 2*N_sites), 
#                   state2, decimal_to_bin_array(state2, 2*N_sites), 
#                   idx, idx2)

            row[c] = idx2
            col[c] = idx
            data[c] = phase_il * phase_ljbar
            c += 1
#             else:# do nothing
#                 print("rejected: ", ni, nj, nibar, njbar)
    # print(c)
    Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.float64)
    Opcsr = Op.tocsr()
    return Opcsr 

def create_3site_spin_flip_hop_ilj_tJmodel(int i, int l, int j, int N_sites, np.ndarray[uint32_t, ndim=1] states):
    # sum_sigma c_i^\dagger_sigma c_j_sigma c_l^\dagger_sigmabar c_l_sigmabar
#     cdef int D, N_sites, c=0
    if i == l or l==j or i==j: return None
    
    cdef uint32_t D, c=0
    cdef uint32_t ibar, lbar, jbar, ni, nl, nj, nibar, nlbar, njbar,
    cdef uint32_t mask_ij, mask_ijbar
    cdef int phase_ij, phase_ijbar,
    cdef uint32_t fermions_in_between_ij, fermions_in_between_ijbar, 
    cdef uint32_t idx=0, idx2=0, state, state2
    
    D = len(states)
    
    row = np.zeros(4*D, dtype = np.uint64)
    col = np.zeros(4*D, dtype = np.uint64)
    data = np.zeros(4*D, dtype = np.float64)
    
    c = 0     # counter
    # i,j for spin down
    # ibar, jbar for spin up
    ibar = i + N_sites  
    lbar = l + N_sites
    jbar = j + N_sites
    
    mask_ij = ((1<<(j+1) - 1) - (1<<(i+1) - 1)) if i < j else ((1<<(i+1) - 1) - (1<<(j+1) - 1))
    mask_ijbar = ((1<<(jbar+1) - 1) - (1<<(ibar+1) - 1)) if i < l else ((1<<(ibar+1) - 1) - (1<<(jbar+1) - 1))
    
#     print(i,l,j)
#     print(ibar, lbar, jbar)
#         , decimal_to_bin_array(mask, 2*N_sites))
    for idx in range(D):
        state = states[idx]
        
        ni = get_bit_j_fast(state, i)
        nl = get_bit_j_fast(state, l)
        nj = get_bit_j_fast(state, j)
        nibar = get_bit_j_fast(state, ibar)
        nlbar = get_bit_j_fast(state, lbar)
        njbar = get_bit_j_fast(state, jbar)
#             print(ni, nj, nibar, njbar)

        if ((nlbar == 1) and (ni == 0 and nj == 1) and (nibar == 0)):
            fermions_in_between_ij = count_1s(state & mask_ij)
            fermions_in_between_ij += 1 if i > j else 0
            
            phase_ij = 1 - 2 * (fermions_in_between_ij % 2)
            state2 = flip_bit2_fast(state, i, j)

            idx2 = find_state(state2, states)

#             print(state, decimal_to_bin_array(state, 2*N_sites), 
#                   state2, decimal_to_bin_array(state2, 2*N_sites), 
#                   idx, idx2)

            row[c] = idx2
            col[c] = idx
            data[c] = phase_ij
            c += 1
        elif ((nl == 1) and (nibar == 0 and njbar == 1) and (ni == 0)):
            fermions_in_between_ijbar = count_1s(state & mask_ijbar)
            
            phase_ijbar = 1 - 2 * (fermions_in_between_ijbar % 2)
            state2 = flip_bit2_fast(state, ibar, jbar)

            idx2 = find_state(state2, states)

#             print(state, decimal_to_bin_array(state, 2*N_sites), 
#                   state2, decimal_to_bin_array(state2, 2*N_sites), 
#                   idx, idx2)

            row[c] = idx2
            col[c] = idx
            data[c] = phase_ijbar
            c += 1
#             else:# do nothing
#                 print("rejected: ", ni, nj, nibar, njbar)
    # print(c)
    Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.float64)
    Opcsr = Op.tocsr()
    return Opcsr