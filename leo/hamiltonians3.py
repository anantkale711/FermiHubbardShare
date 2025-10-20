import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import scipy
import multiprocessing
import leo.cython_files.hop_op_FH as hop_op_FH
import leo.cython_files.hop_op_tJ as hop_op_tJ

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

def generate_N_sector_states_FH(N_sites, N_atoms):
    # N_atoms = #ups - #downs
    dimH = 1<<N_sites
    states = []
    for a in range(dimH):
        n_1s = count_1s(a)
        # zero is no atoms, one is one atom
        if n_1s == N_atoms:
            states.append(a)
    return np.array(states, dtype=np.uint32)

def find_state(s, s_list):
    ''' finds the location of s in s_list using binary search.
    Returns b such that s_list[b] = s '''
    bm = 0
    bM = len(s_list)-1
    while(bm <= bM):
        b = int((bM + bm)/2)
        if s < s_list[b]:
            bM = b-1
        elif s > s_list[b]:
            bm = b+1
        else:
            return b
    #print('State %d not found!'%s)
    return -1
#------------------------------------------------------
# Heisenberg model
def generate_s_sector_states(s, L):
    # s = #ups - #downs
    dimH = 1<<L
    states = []
    for a in range(dimH):
        n_1s = count_1s(a)
        n_0s = L - n_1s
        # zero is up, one is down
        if n_0s - n_1s == s:
            states.append(a)
    return np.array(states)

def create_H_Heisenberg_s_sector(L, s, states):
    dimH = 1<<L
    D = len(states)
    H = np.zeros((D, D))
    for idx in range(D):
        a = states[idx]
        # sigma_+ sigma_- term and sigma_z sigma_z term
        for j in range(L):
            aj = get_bit_j_fast(a, j)
            for k in range(j+1,L):
                ak = get_bit_j_fast(a, k)
                if aj == ak:
                    H[idx, idx] += 1 / 4
                if aj != ak:
                    H[idx, idx] += - 1 / 4
                    b = flip_bit2_fast(a, j, k)
                    idx2 = find_state(b, states)
                    H[idx2, idx] = 1 / 2
    return H


def create_H_Heisenberg_s_sector(connectivity_list, states):
    L = len(connectivity_list)
    D = len(states)
    row = np.zeros(10*L*D, dtype = np.uint64)
    col = np.zeros(10*L*D, dtype = np.uint64)
    data = np.zeros(10*L*D, dtype = np.float64)
    count = 0
    for idx in range(D):
        a = states[idx]
        # sigma_+ sigma_- term and sigma_z sigma_z term
        for j in range(L):
            aj = get_bit_j_fast(a, j)
            for k in connectivity_list[j]:
                if k < j:
                    continue
                ak = get_bit_j_fast(a, k)
                if aj == ak:
                    # H[idx, idx] += 1 / 4
                    row[count], col[count] = idx, idx
                    data[count] = 1/4
                    count += 1
                if aj != ak:
                    # H[idx, idx] += - 1 / 4
                    row[count], col[count] = idx, idx
                    data[count] = -1/4
                    count += 1

                    b = flip_bit2_fast(a, j, k)
                    # idx2 = find_state(b, states)
                    idx2 = hop_op_FH.find_state_long(b, states)
                    # H[idx2, idx] = 1 / 2
                    row[count], col[count] = idx2, idx
                    data[count] = 1/2
                    count += 1
    H = coo_matrix((data[:count+1], (row[:count+1], col[:count+1])), shape = (D, D), dtype = np.float64)
    Hcsr = H.tocsr()
    return Hcsr

#----------------------------------------------
# Fermi-Hubbard model
def create_fermionic_hopping_operator_ij(i, j, states):
    # a_i^\dagger a_j (for spin sigma = 0 or 1)
#     cdef int D, N_sites, c=0
    
    D = len(states)
    
    row = np.zeros(D, dtype = np.uint32)
    col = np.zeros(D, dtype = np.uint32)
    data = np.zeros(D, dtype = np.int8)
    
    c = 0     # counter
    
#     cdef int si, ni, sj, nj, low_s, high_s, spins_in_between, idx1
#     cdef unsigned long tag
    if i == j:
        for idx in range(D):
            state = states[idx]
            #Op[idx, idx] += ni
            row[c] = idx
            col[c] = idx
            data[c] = get_bit_j_fast(state, j)
            c += 1
    else:
        low, high = (i,j) if (i<=j) else (j,i) 
        mask = (1<<(high+1) - 1) - (1<<(low+1) - 1)
#         print(i,j, decimal_to_bin_array(mask, 16))
        for idx in range(D):
            state = states[idx]
            # assume i<=j
            # if i > j, then use transpose, dont recalculate
            
            ni = get_bit_j_fast(state, low)
            nj = get_bit_j_fast(state, high)
            if (ni == 0 and nj == 1):
                fermions_in_between = count_1s(state & mask)
                
                # print(decimal_to_bin_array(state, 16), fermions_in_between)
                
                phase = 1 - 2 * (fermions_in_between % 2)
                state2 = flip_bit2_fast(state, i, j)
                idx2 = find_state(state2, states)
                
                row[c] = idx2
                col[c] = idx
                data[c] = phase
                c += 1
            # else:# do nothing
    # print(c)
    Op = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.int8)
    Opcsr = Op.tocsr()
    return Opcsr 


def create_fermionic_hopping_operators_from_conn_list(states_lists, conn_list):
    N_sites = len(conn_list)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    for sigma in range(2):
        for i in range(N_sites):
            hop_op_list[sigma][i][i] = hop_op_FH.create_fermionic_hopping_operator_ij(i, i, states_lists[sigma])
            
            for j in conn_list[i]:
                if i > j: continue  # only compute for i <= j, use transpose for i > j
                hop_op_list[sigma][i][j] = hop_op_FH.create_fermionic_hopping_operator_ij(i, j, states_lists[sigma])
                if (i != j): hop_op_list[sigma][j][i] = hop_op_list[sigma][i][j].transpose()
            
    return hop_op_list


def create_fermionic_hopping_operators_from_conn_list_fast(states_lists, conn_list, n_processes = None):
    N_sites = len(conn_list)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    
    lst = []
    idx_list = []
    for sigma in range(2):
        for i in range(N_sites):
            lst.append((i, i, states_lists[sigma]))
            idx_list.append((i, i, sigma))

            for j in conn_list[i]:
                if i > j: continue  # only compute for i <= j, use transpose for i > j
                lst.append((i, j, states_lists[sigma]))
                idx_list.append((i, j, sigma))
            
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() - 4
    else:
        n_processes = min(multiprocessing.cpu_count() - 4, n_processes)
        
    p = multiprocessing.Pool(processes = min(n_processes,len(lst)//10))
    output = p.starmap(hop_op_FH.create_fermionic_hopping_operator_ij, lst)

    for count in range(len(output)):
        (i, j, sigma) = idx_list[count]
        hop_op_list[sigma][i][j] = output[count]
        if (i != j): hop_op_list[sigma][j][i] = output[count].transpose()
                
    return hop_op_list


def compute_SzSz_loop(psi, i,j, states_list):
    result = 0
    resi   = 0
    resj   = 0
    states_up = states_list[0]
    states_down = states_list[1]
    psi_mat = psi.reshape(len(states_up), len(states_down))
    for u,su in enumerate(states_up):
        nui = get_bit_j_fast(su, i)
        nuj = get_bit_j_fast(su, j)
        for d,sd in enumerate(states_down):
            ndi = get_bit_j_fast(sd, i)
            ndj = get_bit_j_fast(sd, j)
            result += (nui - ndi)*(nuj-ndj)*np.abs(psi_mat[u,d])**2
            resi   += (nui - ndi)*np.abs(psi_mat[u,d])**2
            resj   += (nuj-ndj)  *np.abs(psi_mat[u,d])**2
    return result - resi * resj

#-------------------------------------------------------------------------------------
# t-J model

def generate_tJ_states(states_up, states_down, N_sites):
    tJ_states=[]
    for su in states_up:
        for sd in states_down:
            if (su & sd) > 0: continue
    #             print("rejected: ", hamiltonians3.decimal_to_bin_array(su,N_sites), hamiltonians3.decimal_to_bin_array(sd,N_sites))
            else:
    #             print(hamiltonians3.decimal_to_bin_array(su,N_sites), hamiltonians3.decimal_to_bin_array(sd,N_sites))
                ns = (1<<(N_sites)) * su + sd
                tJ_states.append(ns)
    # print(len(tJ_states))
    return np.array(tJ_states, dtype=np.uint32)

def create_fermionic_hopping_operators_tJmodel_from_conn_list(tJ_states, conn_list):
    N_sites = len(conn_list)
    hop_op_list = [[[[] for j in range(N_sites)] for i in range(N_sites)] for sigma in range(2)]
    for sigma in range(2):
        for i in range(N_sites):
#             print(i,i)
            hop_op_list[sigma][i][i] = hop_op_tJ.create_hole_hopping_operator_ij_tJmodel(i, i, sigma, N_sites, tJ_states)
            
            for j in conn_list[i]:
                if i > j: continue  # only compute for i <= j, use transpose for i > j
#                 print(i,j)
                hop_op_list[sigma][i][j] = hop_op_tJ.create_hole_hopping_operator_ij_tJmodel(i, j, sigma, N_sites, tJ_states)
                if (i != j): hop_op_list[sigma][j][i] = hop_op_list[sigma][i][j].transpose()
                
    return hop_op_list

def create_H_tJ_hole_hopping_from_conn_list(t_coeff, hop_op_list, conn_list):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns hole-hopping Hamiltonian for t-J model'''
    N_sites = len(conn_list)
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    for sigma in range(2):
        for i in range(N_sites):
            for j in conn_list[i]:  # NN
                if i > j: continue
#                 print(i,j)
                H += -t_coeff * (hop_op_list[sigma][i][j] + hop_op_list[sigma][j][i])
#                 H += -t_coeff * hop_op_list[sigma][i][j]
    return H

def create_H_tJ_electron_exchange_from_conn_list(J_coeff, hop_op_list, states, conn_list):
    ''' Returns electron exchange Hamiltonian for t-J model'''
    # J = 4 t^2/U
    N_sites = len(conn_list)
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    Iden =  scipy.sparse.identity(D)
    for i in range(N_sites):
        for j in conn_list[i]: # NN
            if i<j:
                H += -J_coeff/2 * hop_op_tJ.create_electron_exchange_operator_ij_tJmodel(i, j, N_sites, states)
                for sigma in range(2):
                    H += -J_coeff/2 * (hop_op_list[sigma][i][i] @ hop_op_list[1-sigma][j][j])
    return H

def create_H_tJ_3site_hopping_from_conn_list(J_coeff, hop_op_list, states, conn_list):
    ''' Takes list of hopping operators a_i^\dagger a_j and returns 3site hopping Hamiltonian for t-J model'''
    # J = 4 t^2/U
    N_sites = len(conn_list)
    D = np.shape(hop_op_list[0][0][0])[0]
    H = csr_matrix((D,D), dtype = np.float64)
    for i in range(N_sites):
        for l in conn_list[i]:
            for j in conn_list[l]:
                if (j != i) and j not in conn_list[i]:  # make sure j is really NNN and not NN bc finite size
                    # (i,l,j)
                    H += -J_coeff/4 * hop_op_tJ.create_3site_spin_preserve_hop_ilj_tJmodel(i,l,j, N_sites, states)
                    H += -J_coeff/4 * hop_op_tJ.create_3site_spin_flip_hop_ilj_tJmodel(i,l,j, N_sites, states)
    return H


#-------------------------------------------------------------------------------------
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

def create_connectivity_list_2D_square(N_x, N_y, PBC_x = False, PBC_y = False):
    N_sites = N_x * N_y
    conn_list = [[] for i in range(N_sites)]
    for x in range(N_x):
        for y in range(N_y):
            i = x + N_x * y
            
            if x == 0:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = ((x - 1)%N_x) + N_x * y
                    conn_list[i].append(j)
            elif x == (N_x - 1):
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = ((x + 1)%N_x) + N_x * y
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
                    j = x + N_x * ((y - 1)%N_y)
                    conn_list[i].append(j)
            elif y == (N_y - 1):
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * ((y + 1)%N_y)
                    conn_list[i].append(j)
            else:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
    return conn_list


def create_connectivity_list_2D_triangle(N_x, N_y, PBC_x = False, PBC_y = False):
    N_sites = N_x * N_y
    conn_list = [[] for i in range(N_sites)]
    for x in range(N_x):
        for y in range(N_y):
            i = x + N_x * y
            
            # hopping left-right
            if x == 0:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = ((x - 1)%N_x) + N_x * y
                    conn_list[i].append(j)
            elif x == (N_x - 1):
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = ((x + 1)%N_x) + N_x * y
                    conn_list[i].append(j)
            else:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                
            # hopping up-down (top-right, bottom left)    
            if y == 0:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * ((y - 1)%N_y)
                    conn_list[i].append(j)
            elif y == (N_y - 1):
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * ((y + 1)%N_y)
                    conn_list[i].append(j)
            else:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
            
            
            # Diagonal hopping to the bottom right and top left
            if y == 0:
                if x == 0:
                    if PBC_x == True:
                        j = ((x - 1)%N_x) + N_x * (y+1)
                        conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x+1) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    j = (x - 1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_x == True and PBC_y == True:
                        j = ((x + 1)%N_x) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)
                else:
                    j = (x - 1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x+1) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)                     
            elif y == (N_y - 1):
                if x == 0:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_x == True and PBC_y == True:
                        j = ((x-1)%N_x) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    if PBC_x == True:
                        j = (x+1)%N_x + N_x * (y-1)
                        conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x-1) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                else:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x-1) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
            else:
                if x == 0:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_x == True:
                        j = ((x-1)%N_x) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    j = (x-1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_x == True:
                        j = (x+1)%N_x + N_x * (y-1)
                        conn_list[i].append(j)
                else:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    j = (x-1) + N_x * (y+1)
                    conn_list[i].append(j)
                
    return conn_list

def create_connectivity_list_2D_triangle_XCN(N_x, N_y, PBC_x = False, PBC_y = False):
    N_sites = N_x * N_y
    conn_list = [[] for i in range(N_sites)]
    for x in range(N_x):
        for y in range(N_y):
            i = x + N_x * y
            
            # hopping left-right
            if x == 0:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = ((x - 1)%N_x) + N_x * y
                    conn_list[i].append(j)
            elif x == (N_x - 1):
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = ((x + 1)%N_x) + N_x * y
                    conn_list[i].append(j)
            else:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                
            # hopping up-down (top-right, bottom left)    
            if y == 0:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * ((y - 1)%N_y)
                    conn_list[i].append(j)
            elif y == (N_y - 1):
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * ((y + 1)%N_y)
                    conn_list[i].append(j)
            else:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
            
            
            # Diagonal hopping to the bottom right and top left
            if y == 0:
                if x == 0:
                    if PBC_x == True:
                        j = ((x - 1)%N_x) + N_x * (y+1)
                        conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x+1) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    j = (x - 1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_x == True and PBC_y == True:
                        j = ((x + 1)%N_x) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)
                else:
                    j = (x - 1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x+1) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)                     
            elif y == (N_y - 1):
                if x == 0:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_x == True and PBC_y == True:
                        j = ((x-1)%N_x) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    if PBC_x == True:
                        j = (x+1)%N_x + N_x * (y-1)
                        conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x-1) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                else:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x-1) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
            else:
                if x == 0:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_x == True:
                        j = ((x-1)%N_x) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    j = (x-1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_x == True:
                        j = (x+1)%N_x + N_x * (y-1)
                        conn_list[i].append(j)
                else:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    j = (x-1) + N_x * (y+1)
                    conn_list[i].append(j)
                
    return conn_list

def create_connectivity_list_2D_triangle_YCN(N_x, N_y, PBC_x = False, PBC_y = False):
    N_sites = N_x * N_y
    conn_list = [[] for i in range(N_sites)]
    for x in range(N_x):
        for y in range(N_y):
            i = x + N_x * y
            
            # hopping left-right
            if x == 0:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = ((x - 1)%N_x) + N_x * y
                    conn_list[i].append(j)
            elif x == (N_x - 1):
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                if PBC_x == True:
                    j = ((x + 1)%N_x) + N_x * y
                    conn_list[i].append(j)
            else:
                j = (x + 1) + N_x * y
                conn_list[i].append(j)
                j = (x - 1) + N_x * y
                conn_list[i].append(j)
                
            # hopping up-down (top-right, bottom left)    
            if y == 0:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * ((y - 1)%N_y)
                    conn_list[i].append(j)
            elif y == (N_y - 1):
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
                if PBC_y == True:
                    j = x + N_x * ((y + 1)%N_y)
                    conn_list[i].append(j)
            else:
                j = x + N_x * (y + 1)
                conn_list[i].append(j)
                j = x + N_x * (y - 1)
                conn_list[i].append(j)
            
            
            # Diagonal hopping to the bottom right and top left
            if y == 0:
                if x == 0:
                    if PBC_x == True:
                        j = ((x - 1)%N_x) + N_x * (y+1)
                        conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x+1) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    j = (x - 1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_x == True and PBC_y == True:
                        j = ((x + 1)%N_x) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)
                else:
                    j = (x - 1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x+1) + N_x * ((y - 1)%N_y)
                        conn_list[i].append(j)                     
            elif y == (N_y - 1):
                if x == 0:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_x == True and PBC_y == True:
                        j = ((x-1)%N_x) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    if PBC_x == True:
                        j = (x+1)%N_x + N_x * (y-1)
                        conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x-1) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                else:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_y == True:
                        j = (x-1) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
            else:
                if x == 0:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    if PBC_x == True:
                        j = ((x-1)%N_x) + N_x * ((y+1)%N_y)
                        conn_list[i].append(j)
                elif x == (N_x-1): 
                    j = (x-1) + N_x * (y+1)
                    conn_list[i].append(j)
                    if PBC_x == True:
                        j = (x+1)%N_x + N_x * (y-1)
                        conn_list[i].append(j)
                else:
                    j = (x+1) + N_x * (y-1)
                    conn_list[i].append(j)
                    j = (x-1) + N_x * (y+1)
                    conn_list[i].append(j)
                
    return conn_list
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

def create_H_kin_FH(t, hop_op_list, conn_list):
    H_kin_up   = hop_op_list[0][0][0]
    H_kin_down = hop_op_list[1][0][0]
    N_sites = len(conn_list)
    for i in range(N_sites):
        for j in conn_list[i]:
            if i > j: continue
            #print(i,j)
            H_kin_up   += hop_op_list[0][i][j] + hop_op_list[0][j][i]
            H_kin_down += hop_op_list[1][i][j] + hop_op_list[1][j][i]
    H_kin_up   -= hop_op_list[0][0][0]
    H_kin_down -= hop_op_list[1][0][0]
    return (-t * H_kin_up, -t * H_kin_down)

def create_H_int_FH(U, hop_op_list):
    dimH_up = np.shape(hop_op_list[0][0][0])[0]
    dimH_down = np.shape(hop_op_list[1][0][0])[0]
    H_int = np.zeros((dimH_up, dimH_down))
    temp  = np.zeros((dimH_up, dimH_down))
    N_sites = len(hop_op_list[0])
    for i in range(N_sites):
        np.outer(U * hop_op_list[0][i][i].diagonal(), hop_op_list[1][i][i].diagonal(), out = temp)
        H_int += temp 
    return H_int
