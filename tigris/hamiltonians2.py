import numpy as np
import datetime
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.special import comb

import scipy


import hop_op_FH
import multiprocessing
import hams

def flip_bit_fast(n, i):
    mask = 1<<i
    result = n ^ mask
    return result

def flip_bit2_fast(n, i, j):
    mask = (1<<i) ^ (1<<j)
    result = n ^ mask
    return result

def get_bit_j_fast(n, j):
    temp = n>>j
    mask = 1
    result = temp & mask
    return result

def decimal_to_bin_array(s, L):
    result= np.zeros(L, dtype = np.uint8)
    mask = 1
    n = s
    for i in range(L):
        result[L-1-i] = n & mask
        n = n>>1
    return result

def count_1s(n):
    m = n
    count = 0
    while(m):
        count += m & 1
        m = m >>1
    return count

def get_time():
    t = datetime.datetime.now()
    time_in_seconds = t.hour*3600+ t.minute*60 + t.second + t.microsecond*1e-6
    return time_in_seconds


def tag_state(vec, N_sites):
    sites = np.array(range(1, N_sites + 1))
    return np.sum(np.sqrt(100 * sites + 3) * vec)


def generate_state_vecs_and_tags_BH(N_sites, N_atoms):
    ''' arguments = (N_sites, N_atoms).
    generates vecs[D, N_sites] and tags[D].
    Returns vecs_sorted and tags_sorted where vecs and tags are sorted based on the tags.'''
    D = int(comb(N_atoms + N_sites - 1, N_sites - 1))
    vecs = np.zeros((D, N_sites), dtype = np.int)
    tags = np.zeros(D)
    sites = np.array(range(1, N_sites + 1))
    vec = np.zeros(N_sites, dtype = np.int)
    vec[0] = N_atoms
    vecs[0] = vec
    tags[0] = tag_state(vec, N_sites)
    #print(vec)
    i=1
    while(vec[N_sites-1] < N_atoms):
        for k in range(N_sites-2, -1, -1):
            if vec[k] == 0:
                continue
            break
        #print(k)
        vecs[i,:k]    = vec[:k]
        vecs[i, k]    = vec[k] - 1
        if (k+1 <= N_sites-1):
            #print(N - np.sum(vecs[i,:k+1]))
            vecs[i, k+1]  = N_atoms - np.sum(vecs[i,:k+1])
            vecs[i, k+2:] = 0
        vec = vecs[i]
        tags[i] = tag_state(vec, N_sites)
        #print(i, tags[i], vec)
        i = i + 1
    idx_sorted = tags.argsort()
    tags_sorted = tags[idx_sorted]
    vecs_sorted = vecs[idx_sorted]
    return vecs_sorted, tags_sorted

def find_state_with_tag(tag, tag_list):
    ''' finds the location of state in tag_list using binary search.
    tag_list must be sorted.
    Returns idx such that tag_list[idx] = tag '''
    bm = 0
    bM = len(tag_list)-1
    while(bm <= bM):
        b = int((bM + bm)/2)
        if tag < tag_list[b]:
            bM = b-1
        elif tag > tag_list[b]:
            bm = b+1
        else:
            return b
    print('State with tag %f not found!'%tag)
    return None


def create_hopping_operator_ij(i, j, vecs_sorted, tags_sorted):
    # a_i^\dagger a_j
    D, N_sites = np.shape(vecs_sorted)
    row = np.zeros(D, dtype = np.uint)
    col = np.zeros(D, dtype = np.uint)
    data = np.zeros(D)
    c = 0     # counter
    for idx in range(D):
        if vecs_sorted[idx, j] >= 1:
            if i == j:
                #Op[idx, idx] += vecs_sorted[idx, i]
                row[c] = idx
                col[c] = idx
                data[c] = vecs_sorted[idx, i]
                c += 1
            else:
                vec1 = np.copy(vecs_sorted[idx])
                vec1[i] = vecs_sorted[idx, i] + 1
                vec1[j] = vecs_sorted[idx, j] - 1
                tag = tag_state(vec1, N_sites)
                idx1 = find_state_with_tag(tag, tags_sorted)
                #Op[idx1, idx] += np.sqrt((vecs_sorted[idx, i] + 1)*(vecs_sorted[idx, j]))
                row[c] = idx1
                col[c] = idx
                data[c] = np.sqrt((vecs_sorted[idx, i] + 1)*(vecs_sorted[idx, j]))
                c += 1
    Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.float64)
    Opcsr = Op.tocsr()
    return Opcsr    

def create_hopping_operators(vecs_sorted, tags_sorted):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[] for j in range(N_sites)] for i in range(N_sites)]
    for i in range(N_sites):
        j = i
        hop_op_list[i][j] = create_hopping_operator_ij(i, j, vecs_sorted, tags_sorted)
        if i != N_sites - 1:
            j = i + 1
            hop_op_list[i][j] = create_hopping_operator_ij(i, j, vecs_sorted, tags_sorted)
        if i != 0:
            j = i - 1
            hop_op_list[i][j] = create_hopping_operator_ij(i, j, vecs_sorted, tags_sorted)

    return hop_op_list


def create_H_BH_kin(J_coeff, hop_op_list, PBC = True):
    ''' Takes list of hopping operators a_i^\dagger a_j and return Kinteic Hamiltonian'''
    N_sites = len(hop_op_list[0])
    D = np.shape(hop_op_list[0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    if PBC == True:
        for i in range(N_sites):
            j = (i + 1)%N_sites
            H += -J_coeff * hop_op_list[i][j]
            H += -J_coeff * hop_op_list[j][i]
    else:
        for i in range(N_sites - 1):
            j = i + 1
            H += -J_coeff * hop_op_list[i][j]
            H += -J_coeff * hop_op_list[j][i]
    return H

def create_H_BH_int(U_coeff, hop_op_list):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns interaction Hamiltonian'''
    N_sites = len(hop_op_list[0])
    D = np.shape(hop_op_list[0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    for i in range(N_sites):
        H += U_coeff / 2 * (hop_op_list[i][i] @ hop_op_list[i][i] - hop_op_list[i][i])
    return H

def calculate_rho_SPDM(gs, N_sites, hop_op_list):
    rho_SPDM = np.zeros((N_sites, N_sites), dtype = np.complex)
    for i in range(N_sites):
        for j in range(N_sites):
            rho_SPDM[i,j] = np.vdot(gs, hop_op_list[i][j] @ gs)
    return rho_SPDM



#--------------------------------------------------------
#--------------------------------------------------------
# Fermi-Hubbard


def tag_state_FH(vec, N_sites):
    sites = np.array(range(1, N_sites + 1))
    temp = np.power(4, N_sites - sites, dtype = np.float64)
    return np.sum(temp * vec)


def generate_n_choose_k_vecs(n, k):
    ''' arguments = (n, k).
    generates binary vectors vecs[D, n] where D = nCk.
    '''
    D = int(comb(n,k))
    vecs = np.zeros((D, n), dtype = np.int8)
    if k == 0:
        return vecs
    if ((k < 0) or (k > n)):
        print('Incorrect value of k in nCk')
        return None
    
    vec = np.zeros(n, dtype = np.int8)
    idx  = np.zeros(k, dtype = np.int8)
    for i in range(k):
        idx[i] = i
    vec[idx] = 1
    vecs[0] = vec    
    #print(vec)
    
    i=1
    while(idx[0] <= (n-1) - k):
        vec[:] = 0
        if (idx[k-1] < (n-1)):
            idx[k-1] += 1
            
            vec[idx] = 1
        else:
            for j in range(k-2, -1, -1):
                if (idx[j] >= idx[j+1] - 1):
                    continue
                break
            #print(j)
            idx[j] += 1
            for r in range(j+1, k):
                idx[r] = idx[j] + r - j
            
            vec[idx] = 1
        vecs[i] = vec
        #print(i, vec)
        i = i + 1
    return vecs



def generate_state_vecs_and_tags_FH(N_sites, N_up, N_down):
    ''' arguments = (N_sites, N_up, N_down).
    generates vecs[D, N_sites] and tags[D].
    Returns vecs_sorted and tags_sorted where vecs and tags are sorted based on the tags.'''
    sites = np.array(range(1, N_sites + 1))
    temp = np.power(4, N_sites - sites, dtype = np.float64)
    
    vecs_up = generate_n_choose_k_vecs(N_sites, N_up)
    D_up = np.shape(vecs_up)[0]
    tags_up = np.zeros(D_up, dtype = np.float64)
    for i in range(D_up):
        tags_up[i] = np.sum(temp*(vecs_up[i]).astype(np.float64))

    vecs_down = generate_n_choose_k_vecs(N_sites, N_down)
    D_down = np.shape(vecs_down)[0]
    tags_down = np.zeros(D_down, dtype = np.float64)
    for i in range(D_down):
        tags_down[i] = np.sum(temp*(vecs_down[i]).astype(np.float64))
    
    tags = (np.add.outer(tags_up, 2*tags_down).flatten()).astype(np.float64)

#     # 0-> holon, 1-> up, 2-> down, 3-> doublon
#     for i in range(D_up):
#         for j in range(D_down):
#             c = D_down * i + j
#             vecs[c] = vecs_up[i] + 2*vecs_down[j]
# #             tags[c] = tag_state_FH(vecs[c], N_sites)
#             tags[c] = np.sum(temp * vecs[c])

    vecs_1 = np.repeat(vecs_up, D_down, axis = 0).astype(np.int8)
    vecs_2 = np.tile(2*vecs_down, (D_up, 1)).astype(np.int8)
    vecs = (vecs_1 + vecs_2).astype(np.int8)
    
    idx_sorted = tags.argsort()
    tags_sorted = tags[idx_sorted]
    vecs_sorted = vecs[idx_sorted]
    return vecs_sorted, tags_sorted

def create_fermionic_hopping_operator_ij_sigma(i, j, sigma, vecs_sorted, tags_sorted):
    # a_i^\dagger a_j (for spin sigma = 0 or 1)
#     cdef int D, N_sites, c=0
    
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
    
#     cdef int si, ni, sj, nj, low_s, high_s, spins_in_between, idx1
#     cdef unsigned long tag
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
                    idx1 = hams.find_state_with_tag(tag, tags_sorted)
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


def create_fermionic_hopping_operators(vecs_sorted, tags_sorted):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    for sigma in range(2):
        for i in range(N_sites):
            j = i
            hop_op_list[sigma][i][j] = create_fermionic_hopping_operator_ij_sigma(i, j, sigma, vecs_sorted, tags_sorted)
            j = (i + 1)%N_sites
            hop_op_list[sigma][i][j] = create_fermionic_hopping_operator_ij_sigma(i, j, sigma, vecs_sorted, tags_sorted)
            j = (i - 1)%N_sites
            hop_op_list[sigma][i][j] = create_fermionic_hopping_operator_ij_sigma(i, j, sigma, vecs_sorted, tags_sorted)

    return hop_op_list

def create_fermionic_hopping_operators_onsite_fast(vecs_sorted, tags_sorted):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    for sigma in range(2):
        for i in range(N_sites):
            lst.append((i, i, sigma, vecs_sorted, tags_sorted))
    
    #if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(processes = n_processes)
    output = p.starmap(hop_op_FH.create_fermionic_hopping_operator_ij_sigma, lst)

    count = 0
    for sigma in range(2):
        for i in range(N_sites):
            hop_op_list[sigma][i][i] = output[count]
            count += 1
                        
    return hop_op_list


def create_fermionic_hopping_operators_fast(vecs_sorted, tags_sorted):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    for sigma in range(2):
        for i in range(N_sites):
            lst.append((i, i, sigma, vecs_sorted, tags_sorted))
            lst.append((i, (i + 1)%N_sites, sigma, vecs_sorted, tags_sorted))
            lst.append((i, (i - 1)%N_sites, sigma, vecs_sorted, tags_sorted))
    
    #if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(processes = n_processes)
    output = p.starmap(hop_op_FH.create_fermionic_hopping_operator_ij_sigma, lst)

    count = 0
    for sigma in range(2):
        for i in range(N_sites):
            j = i
            hop_op_list[sigma][i][j] = output[count]
            count += 1
            
            j = (i + 1)%N_sites
            hop_op_list[sigma][i][j] = output[count]
            count += 1
            
            j = (i - 1)%N_sites
            hop_op_list[sigma][i][j] = output[count]
            count += 1
            
    return hop_op_list

def create_fermionic_hopping_operators_all_fast(vecs_sorted, tags_sorted):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    for sigma in range(2):
        for i in range(N_sites):
            for j in range(N_sites):
                lst.append((i, j,sigma, vecs_sorted, tags_sorted))
            
    
    #if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(processes = n_processes)
    output = p.starmap(hop_op_FH.create_fermionic_hopping_operator_ij_sigma, lst)

    count = 0
    for sigma in range(2):
        for i in range(N_sites):
            for j in range(N_sites):
                hop_op_list[sigma][i][j] = output[count]
                count += 1
            
    return hop_op_list

def create_fermionic_hopping_operators_3site_hopping_fast(vecs_sorted, tags_sorted):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    for sigma in range(2):
        for i in range(N_sites):
            lst.append((i, (i + 2)%N_sites, sigma, vecs_sorted, tags_sorted))
            lst.append((i, (i - 2)%N_sites, sigma, vecs_sorted, tags_sorted))
    
    #if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(processes = n_processes)
    output = p.starmap(hop_op_FH.create_fermionic_hopping_operator_ij_sigma, lst)

    count = 0
    for sigma in range(2):
        for i in range(N_sites):
            j = (i + 2)%N_sites
            hop_op_list[sigma][i][j] = output[count]
            count += 1
            
            j = (i - 2)%N_sites
            hop_op_list[sigma][i][j] = output[count]
            count += 1
            
    return hop_op_list


def create_fermionic_hopping_operators_2D_fast(vecs_sorted, tags_sorted, N_x, N_y):
    '''create_fermionic_hopping_operators_2D_fast(vecs_sorted, tags_sorted, N_x, N_y):'''
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    # index = x + N_x * y
    # x = index % N_x, y = (index - x)/N_x
    
    lst = []
    for sigma in range(2):
        for x in range(N_x):
            for y in range(N_y):
                i = x + N_x * y
                lst.append((i, i, sigma, vecs_sorted, tags_sorted))
                j = (x+1) % N_x + N_x * y
                lst.append((i, j, sigma, vecs_sorted, tags_sorted))
                j = (x-1) % N_x + N_x * y
                lst.append((i, j, sigma, vecs_sorted, tags_sorted))
                j = x + N_x * ((y+1)%N_y)
                lst.append((i, j, sigma, vecs_sorted, tags_sorted))
                j = x + N_x * ((y-1)%N_y)
                lst.append((i, j, sigma, vecs_sorted, tags_sorted))

    #if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count() - 2 
    p = multiprocessing.Pool(processes = n_processes)
    output = p.starmap(hop_op_FH.create_fermionic_hopping_operator_ij_sigma, lst)

    count = 0
    for sigma in range(2):
        for x in range(N_x):
            for y in range(N_y):
                i = x + N_x * y
                j = i
                hop_op_list[sigma][i][j] = output[count]
                count += 1
            
                j = (x+1) % N_x + N_x * y
                hop_op_list[sigma][i][j] = output[count]
                count += 1
                
                j = (x-1) % N_x + N_x * y
                hop_op_list[sigma][i][j] = output[count]
                count += 1
                
                j = x + N_x * ((y+1)%N_y)
                hop_op_list[sigma][i][j] = output[count]
                count += 1
                
                j = x + N_x * ((y-1)%N_y)
                hop_op_list[sigma][i][j] = output[count]
                count += 1
    return hop_op_list

    
def create_fermionic_hopping_operators_from_conn_list_fast(vecs_sorted, tags_sorted, conn_list):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    idx_list = []
    for sigma in range(2):
        for i in range(N_sites):
            lst.append((i, i, sigma, vecs_sorted, tags_sorted))
            idx_list.append((i, i, sigma))

            for j in conn_list[i]:
                lst.append((i, j, sigma, vecs_sorted, tags_sorted))
                idx_list.append((i, j, sigma))
            
    #if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(processes = n_processes)
    output = p.starmap(hop_op_FH.create_fermionic_hopping_operator_ij_sigma, lst)

    for count in range(len(output)):
        (i, j, sigma) = idx_list[count]
        hop_op_list[sigma][i][j] = output[count]
                
    return hop_op_list

def create_fermionic_hopping_operators_from_conn_list(vecs_sorted, tags_sorted, conn_list):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    idx_list = []
    for sigma in range(2):
        for i in range(N_sites):
            hop_op_list[sigma][i][i] = hop_op_FH.create_fermionic_hopping_operator_ij_sigma(i, i, sigma, vecs_sorted, tags_sorted)
            for j in conn_list[i]:
                hop_op_list[sigma][i][j] = hop_op_FH.create_fermionic_hopping_operator_ij_sigma(i, j, sigma, vecs_sorted, tags_sorted)
                
    return hop_op_list

def create_fermionic_hopping_operators_3site_hopping_from_conn_list_fast(vecs_sorted, tags_sorted, conn_list):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    count = 0
    idx_list = []
    for sigma in range(2):
        for i in range(N_sites):
            for l in conn_list[i]:
                for j in conn_list[l]:
                    if (j != i) and j not in conn_list[i]:  # make sure j is really NNN and not NN bc finite size
                        # (i,l,j)
                        idx_list.append((i,j,sigma))
                        lst.append((i, j, sigma, vecs_sorted, tags_sorted))
             
    #if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(processes = n_processes)
    output = p.starmap(hop_op_FH.create_fermionic_hopping_operator_ij_sigma, lst)

    for count in range(len(output)):
        (i, j, sigma) = idx_list[count]
        hop_op_list[sigma][i][j] = output[count]
                
    return hop_op_list

def new_create_fermionic_hopping_operators_fast(vecs_sorted, tags_sorted):
    D, N_sites = np.shape(vecs_sorted)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    for sigma in range(2):
        for i in range(N_sites):
            lst.append((i, i, sigma, sigma, vecs_sorted, tags_sorted))
            lst.append((i, (i + 1)%N_sites, sigma, sigma, vecs_sorted, tags_sorted))
            lst.append((i, (i - 1)%N_sites, sigma, sigma, vecs_sorted, tags_sorted))
    
    #if __name__ == '__main__':
    n_processes = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(processes = n_processes)
    output = p.starmap(hop_op_FH.create_fermionic_operator_mixed, lst)

    count = 0
    for sigma in range(2):
        for i in range(N_sites):
            j = i
            hop_op_list[sigma][i][j] = output[count]
            count += 1
            
            j = (i + 1)%N_sites
            hop_op_list[sigma][i][j] = output[count]
            count += 1
            
            j = (i - 1)%N_sites
            hop_op_list[sigma][i][j] = output[count]
            count += 1
            
    return hop_op_list

def create_H_FH_kin(t_coeff, hop_op_list, PBC = False):
    ''' Takes list of hopping operators a_i^\dagger a_j and return Kinteic Hamiltonian'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.int8)
    if PBC == False:
        for sigma in range(2):
            for i in range(N_sites - 1):
                j = i + 1
                H += -t_coeff * hop_op_list[sigma][i][j]
                H += -t_coeff * hop_op_list[sigma][j][i]
    else:
        for sigma in range(2):
            for i in range(N_sites):
                j = (i + 1)%N_sites
                H += -t_coeff * hop_op_list[sigma][i][j]
                H += -t_coeff * hop_op_list[sigma][j][i]
    return H


def create_H_FH_kin_2D(t_coeff, hop_op_list, N_x, N_y, PBC_x = False, PBC_y = False):
    ''' Takes list of hopping operators a_i^\dagger a_j and return Kinteic Hamiltonian'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.int8)
    if PBC_x == False:
        for sigma in range(2):
            for x in range(N_x - 1):
                for y in range(N_y):
                    i = x + N_x * y
                    j = (x + 1)  + N_x * y
                    H += -t_coeff * hop_op_list[sigma][i][j]
                    H += -t_coeff * hop_op_list[sigma][j][i]
    else:
        for sigma in range(2):
            for x in range(N_x):
                for y in range(N_y):
                    i = x + N_x * y
                    j = (x+1) % N_x + N_x * y
                    H += -t_coeff * hop_op_list[sigma][i][j]
                    H += -t_coeff * hop_op_list[sigma][j][i]
                    
    if PBC_y == False:
        for sigma in range(2):
            for x in range(N_x):
                for y in range(N_y - 1):
                    i = x + N_x * y
                    j = x  + N_x * (y + 1)
                    H += -t_coeff * hop_op_list[sigma][i][j]
                    H += -t_coeff * hop_op_list[sigma][j][i]
    else:
        for sigma in range(2):
            for x in range(N_x):
                for y in range(N_y):
                    i = x + N_x * y
                    j = x  + N_x * ((y + 1) % N_y)
                    H += -t_coeff * hop_op_list[sigma][i][j]
                    H += -t_coeff * hop_op_list[sigma][j][i]
    
    return H

def create_H_FH_int(U_coeff, hop_op_list):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns interaction Hamiltonian'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.int8)
    for i in range(N_sites):
        H += U_coeff * (hop_op_list[0][i][i] @ hop_op_list[1][i][i])
    return H


#------------------------------------------
# t-J model
#------------------------------------------


def create_H_tJ_hole_hopping(t_coeff, hop_op_list, PBC = False):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns hole-hopping Hamiltonian for t-J model'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.int8)
    Iden = scipy.sparse.identity(D)
    if PBC == False:
        for sigma in range(2):
            for i in range(N_sites - 1):
                j = i + 1
                H += -t_coeff * hop_op_list[sigma][i][j] @ (Iden-hop_op_list[1-sigma][i][i]) @ (Iden-hop_op_list[1-sigma][j][j])
                H += -t_coeff * hop_op_list[sigma][j][i] @ (Iden-hop_op_list[1-sigma][i][i]) @ (Iden-hop_op_list[1-sigma][j][j])
    else:
        for sigma in range(2):
            for i in range(N_sites):
                j = (i + 1)%N_sites
                H += -t_coeff * hop_op_list[sigma][i][j] @ (Iden-hop_op_list[1-sigma][i][i]) @ (Iden-hop_op_list[1-sigma][j][j])
                H += -t_coeff * hop_op_list[sigma][j][i] @ (Iden-hop_op_list[1-sigma][i][i]) @ (Iden-hop_op_list[1-sigma][j][j])
    return H


def create_H_tJ_electron_exchange(J_coeff, hop_op_list, PBC = False):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns electron exchange Hamiltonian for t-J model
    J = 2 t^2/U 
    '''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    Iden = scipy.sparse.identity(D)
    if PBC == False:
        for sigma in range(2):
            for i in range(N_sites - 1):
                j = i + 1
                H += -J_coeff/2 * hop_op_list[sigma][i][j] @ hop_op_list[1-sigma][j][i] 
                H += -J_coeff/2 * hop_op_list[sigma][j][i] @ hop_op_list[1-sigma][i][j]
                
                H += -J_coeff/2 * hop_op_list[sigma][i][i] @ hop_op_list[1-sigma][j][j] @ \
                                (Iden - hop_op_list[1-sigma][i][i]) @ (Iden - hop_op_list[sigma][j][j])
                H += -J_coeff/2 * hop_op_list[sigma][j][j] @ hop_op_list[1-sigma][i][i] @ \
                                (Iden - hop_op_list[1-sigma][j][j]) @ (Iden - hop_op_list[sigma][i][i])
    else:       
        for sigma in range(2):
            for i in range(N_sites):
                j = (i + 1)%N_sites
                H += -J_coeff/2 * hop_op_list[sigma][i][j] @ hop_op_list[1-sigma][j][i] 
                H += -J_coeff/2 * hop_op_list[sigma][j][i] @ hop_op_list[1-sigma][i][j]
                
                H += -J_coeff/2 * hop_op_list[sigma][i][i] @ hop_op_list[1-sigma][j][j] @ \
                                (Iden - hop_op_list[1-sigma][i][i]) @ (Iden - hop_op_list[sigma][j][j])
                H += -J_coeff/2 * hop_op_list[sigma][j][j] @ hop_op_list[1-sigma][i][i] @ \
                                (Iden - hop_op_list[1-sigma][j][j]) @ (Iden - hop_op_list[sigma][i][i])
                
    return H

def create_H_tJ_3site_hopping(J_coeff, hop_op_list, PBC = False):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns 3site hopping Hamiltonian for t-J model'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    Iden = scipy.sparse.identity(D)
    if PBC == False:
        for sigma in range(2):
            for i in range(N_sites - 2):
                l = i + 1
                j = l + 1
                
                #(i,l,j)
                H += -J_coeff/2 * hop_op_list[1-sigma][i][l] @ hop_op_list[sigma][l][j] @ \
                                (Iden - hop_op_list[sigma][i][i]) @ (Iden - hop_op_list[1-sigma][j][j])
                #(j,l,i)
                H += -J_coeff/2 * hop_op_list[1-sigma][j][l] @ hop_op_list[sigma][l][i] @ \
                                (Iden - hop_op_list[sigma][j][j]) @ (Iden - hop_op_list[1-sigma][i][i])
                
                #(i,l,j)
                H += -J_coeff/2 * hop_op_list[sigma][i][j] @ hop_op_list[1-sigma][l][l] @ \
                                (Iden - hop_op_list[sigma][l][l]) @ (Iden - hop_op_list[1-sigma][i][i]) @ \
                                (Iden - hop_op_list[1-sigma][j][j])
                #(j,l,i)
                H += -J_coeff/2 * hop_op_list[sigma][j][i] @ hop_op_list[1-sigma][l][l] @ \
                                (Iden - hop_op_list[sigma][l][l]) @ (Iden - hop_op_list[1-sigma][j][j]) @ \
                                (Iden - hop_op_list[1-sigma][i][i])
    else:
        for sigma in range(2):
            for i in range(N_sites):
                l = (i + 1) % N_sites
                j = (l + 1) % N_sites
                
                #(i,l,j)
                H += -J_coeff/2 * hop_op_list[1-sigma][i][l] @ hop_op_list[sigma][l][j] @ \
                                (Iden - hop_op_list[sigma][i][i]) @ (Iden - hop_op_list[1-sigma][j][j])
                #(j,l,i)
                H += -J_coeff/2 * hop_op_list[1-sigma][j][l] @ hop_op_list[sigma][l][i] @ \
                                (Iden - hop_op_list[sigma][j][j]) @ (Iden - hop_op_list[1-sigma][i][i])
                
                #(i,l,j)
                H += -J_coeff/2 * hop_op_list[sigma][i][j] @ hop_op_list[1-sigma][l][l] @ \
                                (Iden - hop_op_list[sigma][l][l]) @ (Iden - hop_op_list[1-sigma][i][i]) @ \
                                (Iden - hop_op_list[1-sigma][j][j])
                #(j,l,i)
                H += -J_coeff/2 * hop_op_list[sigma][j][i] @ hop_op_list[1-sigma][l][l] @ \
                                (Iden - hop_op_list[sigma][l][l]) @ (Iden - hop_op_list[1-sigma][j][j]) @ \
                                (Iden - hop_op_list[1-sigma][i][i])
                
    return H

# ------------------------------------------------

def create_connectivity_list_1D(N_sites, PBC = False):
    conn_list = [[] for i in range(N_sites)]
    for i in range(N_sites):
        if i == 0:
            conn_list[i].append(i+1)
            if PBC == True:
                conn_list[i].append((i-1)%N_sites)
        elif i == (N_sites - 1):
            conn_list[i].append(i-1)
            if PBC == True:
                conn_list[i].append((i+1)%N_sites)
        else:
            conn_list[i].append(i-1)
            conn_list[i].append(i+1)
    return conn_list

def create_connectivity_list_2D(N_x, N_y, PBC_x = False, PBC_y = False):
    N_sites = N_x * N_y
    conn_list = [[] for i in range(N_sites)]
    for x in range(N_x):
        for y in range(N_y):
            i = x + N_x * y
            
            if x == 0:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = (x - 1)%N_x + N_x * y
                    conn_list[i].append(j)
            elif x == (N_x - 1):
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = (x + 1)%N_x + N_x * y
                    conn_list[i].append(j)
            else:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                
                
            if y == 0:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * (y - 1)%N_y
                    conn_list[i].append(j)
            elif y == (N_y - 1):
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * (y + 1)%N_y
                    conn_list[i].append(j)
            else:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
    return conn_list


def create_H_FH_kin_from_conn_list(t_coeff, hop_op_list, conn_list):
    ''' Takes list of hopping operators a_i^\dagger a_j, connectivity_list and returns Kinetic Hamiltonian'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.int8)
    for sigma in range(2):
        for i in range(N_sites):
            for j in conn_list[i]:
                # j is NN
                H += -t_coeff * hop_op_list[sigma][i][j]
    return H



def create_H_tJ_hole_hopping_from_conn_list(t_coeff, hop_op_list, conn_list):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns hole-hopping Hamiltonian for t-J model'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.int8)
    Iden = scipy.sparse.identity(D)
    for sigma in range(2):
        for i in range(N_sites):
            for j in conn_list[i]:  # NN
                H += -t_coeff * hop_op_list[sigma][i][j] @ (Iden-hop_op_list[1-sigma][i][i]) @ (Iden-hop_op_list[1-sigma][j][j])
    return H

def create_H_tJ_electron_exchange_from_conn_list(J_coeff, hop_op_list, conn_list):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns electron exchange Hamiltonian for t-J model'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    Iden = scipy.sparse.identity(D)
    for sigma in range(2):
        for i in range(N_sites):
            for j in conn_list[i]: # NN
                H += -J_coeff/2 * hop_op_list[sigma][i][j] @ hop_op_list[1-sigma][j][i] 
                
                H += -J_coeff/2 * hop_op_list[sigma][i][i] @ hop_op_list[1-sigma][j][j] @ \
                                (Iden - hop_op_list[1-sigma][i][i]) @ (Iden - hop_op_list[sigma][j][j])
                
    return H

def create_H_tJ_3site_hopping_from_conn_list(J_coeff, hop_op_list, conn_list):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns 3site hopping Hamiltonian for t-J model'''
    N_sites = len(hop_op_list[0][0])
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    Iden = scipy.sparse.identity(D)
    for sigma in range(2):
        for i in range(N_sites):
            for l in conn_list[i]:
                for j in conn_list[l]:
                    if (j != i) and j not in conn_list[i]:  # make sure j is really NNN and not NN bc finite size
                        # (i,l,j)
                        H += -J_coeff/2 * hop_op_list[1-sigma][i][l] @ hop_op_list[sigma][l][j] @ \
                                        (Iden - hop_op_list[sigma][i][i]) @ (Iden - hop_op_list[1-sigma][j][j])
                        
                        H += -J_coeff/2 * hop_op_list[sigma][i][j] @ hop_op_list[1-sigma][l][l] @ \
                                        (Iden - hop_op_list[sigma][l][l]) @ (Iden - hop_op_list[1-sigma][i][i]) @ \
                                        (Iden - hop_op_list[1-sigma][j][j])
    return H


#------------------------------------------------------