import numpy as np
import datetime
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

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
    print('State %d not found!'%s)
    return None


#------------------------------------------------------
# Heisenberg model

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
    print('State %d not found!'%s)
    return None

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
    row = np.zeros(10*D, dtype = np.uint32)
    col = np.zeros(10*D, dtype = np.uint32)
    data = np.zeros(10*D, dtype = np.int8)
    count = 0
    for idx in range(D):
        a = states[idx]
        # sigma_+ sigma_- term and sigma_z sigma_z term
        for j in range(L):
            aj = get_bit_j_fast(a, j)
            for k in connectivity_list[j]:
                ak = get_bit_j_fast(a, k)
                if aj == ak:
                    # H[idx, idx] += 1 / 4
                    row[count], col[count] = idx, idx
                    data[count] = 1
                    count += 1
                if aj != ak:
                    # H[idx, idx] += - 1 / 4
                    row[count], col[count] = idx, idx
                    data[count] = -1
                    count += 1

                    b = flip_bit2_fast(a, j, k)
                    idx2 = find_state(b, states)
                    # idx2 = hams.find_state_with_tag(b, states)
                    # H[idx2, idx] = 1 / 2
                    row[count], col[count] = idx2, idx
                    data[count] = 2
                    count += 1
    H = coo_matrix((data[:count+1], (row[:count+1], col[:count+1])), shape = (D, D), dtype = np.int8)
    Hcsr = H.tocsr()
    return Hcsr



#----------------------------------------------------------------------
# Trapped Ions

# $$ H = \sum_{i,j}{\frac{J}{|i-j|^\alpha} \sigma^x_i \sigma^x_j} + B \sum_i{\sigma^z_i} $$

def create_H_ising_dense(J, Bz, alpha, L):
    dimH = 1<<L
    H = np.zeros((dimH, dimH))
    for a in range(dimH):
        # sigma_z terms    
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[a,a] += Bz * (1 - 2*aj) 
        
        #sigma_x sigma_x terms
        for j in range(L):
            for k in range(j+1,L):
                J_jk = J/(abs(j-k)**alpha) 
                b = flip_bit2_fast(a, j, k)
                H[b,a] += J_jk
    return H

def generate_even_sector_states(L):
    dimH = 1<<L
    states = []
    for a in range(dimH):
        if count_1s(a) % 2 == 0:
            states.append(a)
    return np.array(states)

def generate_odd_sector_states(L):
    dimH = 1<<L
    states = []
    for a in range(dimH):
        if count_1s(a) % 2 == 1:
            states.append(a)
    return np.array(states)

def create_H_ising_dense_sector(J, Bz, alpha, L, states):
    dimH = 1<<L
    D = len(states)
    H = np.zeros((D, D))
    for idx in range(D):
        a = states[idx]
        # sigma_z terms    
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[idx, idx] += Bz * (1 - 2*aj) 
        
        #sigma_x sigma_x terms
        for j in range(L):
            for k in range(j+1,L):
                J_jk = J/(abs(j-k)**alpha) 
                b = flip_bit2_fast(a, j, k)
                idx2 = find_state(b, states)
                H[idx2, idx] += J_jk
    return H

# Trapped Ion with no Ising symmetry
def create_H_ising_modified(J, Bz, Bx, alpha, L):
    dimH = 1<<L
    H = np.zeros((dimH, dimH))
    for a in range(dimH):
        # sigma_z terms    
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[a,a] += Bz * (1 - 2*aj) 
        
        # sigma_x terms    
        for j in range(L-1):
            b = flip_bit_fast(a,j)
            H[b,a] += Bx 
        
        #sigma_x sigma_x terms
        for j in range(L):
            for k in range(j+1,L):
                J_jk = J/(abs(j-k)**alpha) 
                b = flip_bit2_fast(a, j, k)
                H[b,a] += J_jk
    return H

# Trapped Ion with no Ising symmetry and slight disorder
def create_H_ising_disorder_modified(J, Bz, Bx, alpha, W, L):
    dimH = 1<<L
    H = np.zeros((dimH, dimH))
    h_list = np.random.uniform(-W, W, size = L)
    for a in range(dimH):
        # sigma_z terms    
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[a,a] += Bz * (1 - 2*aj) 
        
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[a,a] += h_list[j] * (1 - 2*aj) 
        
        # sigma_x terms    
        for j in range(L-1):
            b = flip_bit_fast(a,j)
            H[b,a] += Bx 
        
        #sigma_x sigma_x terms
        for j in range(L):
            for k in range(j+1,L):
                J_jk = J/(abs(j-k)**alpha) 
                b = flip_bit2_fast(a, j, k)
                H[b,a] += J_jk
    return H

# def create_H_ising_disorder_modified_sparse(J, Bz, Bx, alpha, W, L):
#     dimH = 1<<L
#     H = dok_matrix((dimH, dimH), dtype = np.float64)
#     h_list = np.random.uniform(-W, W, size = L)
#     for a in range(dimH):
#         # sigma_z terms    
#         for j in range(L):
#             aj = get_bit_j_fast(a,j)
#             H[a,a] = H[a,a] + Bz * (1 - 2*aj) 
        
#         for j in range(L):
#             aj = get_bit_j_fast(a,j)
#             H[a,a] = H[a,a] + h_list[j] * (1 - 2*aj) 
        
#         # sigma_x terms    
#         for j in range(L-1):
#             b = flip_bit_fast(a,j)
#             H[b,a] = H[b,a] + Bx 
        
#         #sigma_x sigma_x terms
#         for j in range(L):
#             for k in range(j+1,L):
#                 J_jk = J/(abs(j-k)**alpha) 
#                 b = flip_bit2_fast(a, j, k)
#                 H[b,a] = H[b,a] + J_jk
#     Hcoo = H.tocoo()
#     Hcsr = Hcoo.tocsr()
#     return Hcsr

def create_H_ising_disorder_modified_sparse(J, Bz, Bx, alpha, W, L):
    dimH = 1<<L
    row  = np.zeros(dimH * L**2 * 10, dtype = np.uint32)
    col  = np.zeros(dimH * L**2 * 10, dtype = np.uint32)
    data = np.zeros(dimH * L**2 * 10, dtype = np.float64)
    c = 0
    h_list = np.random.uniform(-W, W, size = L)
    for a in range(dimH):
        # sigma_z terms    
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            row[c] = a
            col[c] = a
            data[c] = Bz * (1 - 2*aj) # H[a,a] += Bz * (1 - 2*aj) 
            c = c+1
            
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            row[c]  = a
            col[c]  = a
            data[c] = h_list[j] * (1 - 2*aj)  # H[a,a] += h_list[j] * (1 - 2*aj) 
            c = c + 1
            
        # sigma_x terms    
        for j in range(L-1):
            b = flip_bit_fast(a,j)
            row[c]  = b
            col[c]  = a
            data[c] = Bx   # H[b,a] += Bx 
            c = c + 1
            
        #sigma_x sigma_x terms
        for j in range(L):
            for k in range(j+1,L):
                J_jk = J/(abs(j-k)**alpha) 
                b = flip_bit2_fast(a, j, k)
                row[c]  = b
                col[c]  = a
                data[c] = J_jk  # H[b,a] += J_jk
                c = c + 1
    H = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (dimH, dimH), dtype = np.float64)
    Hcsr = H.tocsr()
    return Hcsr

def create_H_non_interacting(W, L):
    dimH = 1<<L
    H = np.zeros((dimH, dimH))
    Bx_list = np.random.uniform(-W, W, size = L)
    for a in range(dimH):
        # sigma_z terms    
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[a,a] += (1 - 2*aj) 
    # sigma_x terms    
        for j in range(L-1):
            b = flip_bit_fast(a,j)
            H[b,a] += Bx_list[j] 
    return H
#-----------------------------------------------------------------------
# Trapped Ion sector-resolved (Bz >> J)
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

def create_H_flip_flop_s_sector(J, Bz, alpha, L, s, states):
    dimH = 1<<L
    D = len(states)
    H = np.zeros((D, D))
    for idx in range(D):
        a = states[idx]
        # Bz term
        H[idx, idx] += s * Bz
        # sigma_+ sigma_- term and (pertubative) sigma_z sigma_z term
        for j in range(L):
            aj = get_bit_j_fast(a, j)
            for k in range(j+1,L):
                ak = get_bit_j_fast(a, k)
                J_jk = J/(abs(j-k)**alpha) 
                # if aj == ak:
                    # H[idx, idx] += J_jk**2 / (4 * Bz)
                if aj != ak:
                    # H[idx, idx] += - J_jk**2 / (4 * Bz)
                    b = flip_bit2_fast(a, j, k)
                    idx2 = find_state(b, states)
                    H[idx2, idx] += J_jk
    return H

def create_H_flip_flop_with_disorder_s_sector(J, Bz, alpha, W, L, s, states):
    dimH = 1<<L
    D = len(states)
    H = np.zeros((D, D))
    h_list = np.random.uniform(-W, W, size = L)
    for idx in range(D):
        a = states[idx]
        # Bz term
        H[idx, idx] += s * Bz
        # disorder sigma_z term
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[idx,idx] += h_list[j] * (1 - 2*aj)     
        # sigma_+ sigma_- term and (pertubative) sigma_z sigma_z term
        for j in range(L):
            aj = get_bit_j_fast(a, j)
            for k in range(j+1,L):
                ak = get_bit_j_fast(a, k)
                J_jk = J/(abs(j-k)**alpha) 
                #if aj == ak:
                #    H[idx, idx] += J_jk**2 / (4 * Bz)
                if aj != ak:
                    #H[idx, idx] += - J_jk**2 / (4 * Bz)
                    b = flip_bit2_fast(a, j, k)
                    idx2 = find_state(b, states)
                    H[idx2, idx] += J_jk
    return H

def create_H_flip_flop_with_disorder_s_sector_sparse(J, Bz, alpha, W, L, s, states):
    dimH = 1<<L
    D = len(states)
    H = dok_matrix((D, D), dtype = np.float64)
    h_list = np.random.uniform(-W, W, size = L)
    for idx in range(D):
        a = states[idx]
        # Bz term
        H[idx, idx] += s * Bz
        # disorder sigma_z term
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[idx,idx] += h_list[j] * (1 - 2*aj)     
        # sigma_+ sigma_- term and (pertubative) sigma_z sigma_z term
        for j in range(L):
            aj = get_bit_j_fast(a, j)
            for k in range(j+1,L):
                ak = get_bit_j_fast(a, k)
                J_jk = J/(abs(j-k)**alpha) 
                #if aj == ak:
                #    H[idx, idx] += J_jk**2 / (4 * Bz)
                if aj != ak:
                    #H[idx, idx] += - J_jk**2 / (4 * Bz)
                    b = flip_bit2_fast(a, j, k)
                    idx2 = find_state(b, states)
                    H[idx2, idx] += J_jk
    return H.tocsr()

#------------------------------------------------------------------------
# Sparse local Pauli matrices
def create_X_j(j, L):
    dimH = 1<<L
    Op = dok_matrix((dimH, dimH), dtype = np.float64)
    for a in range(dimH):
        # X term
        aj = get_bit_j_fast(a, j)
        b = flip_bit_fast(a, j)
        Op[b, a] = Op[b, a] + 1
    Op_csr = Op.tocsr()
    return Op_csr

def create_Z_j(j, L):
    dimH = 1<<L
    Op = dok_matrix((dimH, dimH), dtype = np.float64)
    for a in range(dimH):
        aj = get_bit_j_fast(a, j)
        Op[a, a] = Op[a, a] + (1 - 2*aj)
    Op_csr = Op.tocsr()
    return Op_csr

def create_Pauli_lists(L):
    X_list = [[] for i in range(L)]
    Z_list = [[] for i in range(L)]
    Y_list = [[] for i in range(L)]
    for i in range(L):
        X_list[i] = create_X_j(i, L)
        Z_list[i] = create_Z_j(i, L)
        Y_list[i] = 1j * X_list[i] @ Z_list[i]
    return (X_list, Y_list, Z_list)

#----------------------------------------------------------------------
# GUE Hamiltonians
def create_H_GUE(dimH):
    A = 1/np.sqrt(2)* ( np.random.normal(0, 1/np.sqrt(dimH), size = (dimH, dimH)) \
                       + 1j * np.random.normal(0, 1/np.sqrt(dimH), size = (dimH, dimH)) )
    H = (A + np.conjugate(A.transpose())) /np.sqrt(2)
    return H

def create_H_GUE_unnormalized(dimH):
    A = 1/np.sqrt(2)* ( np.random.normal(0, 1, size = (dimH, dimH)) \
                       + 1j * np.random.normal(0, 1, size = (dimH, dimH)) )
    H = (A + np.conjugate(A.transpose())) /np.sqrt(2)
    return H

def create_H_GOE(dimH):
    A = np.random.normal(0, 1/np.sqrt(dimH), size = (dimH, dimH))
    H = (A + A.transpose()) /np.sqrt(2)
    return H

def create_H_GOE_unnormalized(dimH):
    A = np.random.normal(0, 1, size = (dimH, dimH))
    H = (A + A.transpose()) /np.sqrt(2)
    return H
#------------------------------------------------------------------------
# Random all-to-all 2-body 
def create_H_all_to_all_2_body(W, L):
    dimH = 1<<L
    H = np.zeros((dimH, dimH))
    B_list = np.random.uniform(-W, W, size = L)
    Jx_list = np.random.uniform(-W, W, size = (L,L))
    Jz_list = np.random.uniform(-W, W, size = (L,L))
    for a in range(dimH):
        # sigma_x terms    
        for j in range(L):
            b = flip_bit_fast(a,j)
            H[b,a] += 1
            
        # sigma_z terms    
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[a,a] += B_list[j] * (1 - 2*aj) 
        
        # sigma_z sigma_z terms    
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            for k in range(j+1, L):
                ak = get_bit_j_fast(a,k)
                H[a,a] += Jz_list[j, k] * (1 - 2*aj) * (1 - 2*ak) 
        
        #sigma_x sigma_x terms
        for j in range(L):
            for k in range(j+1,L):
                b = flip_bit2_fast(a, j, k)
                H[b,a] += Jx_list[j,k]
    return H 

def create_H_local_2_body(W, L):
    dimH = 1<<L
    H = np.zeros((dimH, dimH))
    B_list = np.random.uniform(-W, W, size = L)
    Jx_list = np.random.uniform(-W, W, size = L)
    Jz_list = np.random.uniform(-W, W, size = L)
    for a in range(dimH):
        # sigma_x terms    
        for j in range(L):
            b = flip_bit_fast(a,j)
            H[b,a] += 1
            
        # sigma_z terms
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[a,a] += B_list[j] * (1 - 2*aj) 
        
        # sigma_z sigma_z terms
        for j in range(L-1):
            aj = get_bit_j_fast(a,j)
            aj1 = get_bit_j_fast(a,j+1)
            H[a,a] += Jz_list[j] * (1 - 2*aj) * (1 - 2*aj1) 
        
        #sigma_x sigma_x terms
        for j in range(L-1):
            b = flip_bit2_fast(a, j, j+1)
            H[b,a] += Jx_list[j]
    return H 

def create_H_local_GUE(L):
    dimH = 1<<L
    H = np.zeros((dimH, dimH))
    return H

#-------------------------------------------------------------
# PXP Hamiltonians

def check_blockade_violation(s, L):
    bin_array = decimal_to_bin_array(s, L)
    for i in range(L):
        if i == 0:
            if bin_array[i] and bin_array[i+1]:
                return True
        elif i == L-1:
            if bin_array[i-1] and bin_array[i]:
                return True
        else:
            if (bin_array[i-1] and bin_array[i]) or (bin_array[i] and bin_array[i+1]):
                return True
    return False

def generate_blockade_sector_states(L):
    dimH  = 1<<L
    s_list = []
    for a in range(dimH):
        if check_blockade_violation(a,L) == False:
            s_list.append(a)
    s_array = np.array(s_list)
    return s_array


def check_blockade_violation_fast(s):
    r = s << 1
    t = s >> 1
    if (s & r) or (s & t):
        return True
    return False

def generate_blockade_sector_states_fast(L):
    dimH  = 1<<L
    D1 = int(1.62**L) * 2
    s_list = np.zeros(D1, dtype = np.uint32)
    idx = 0
    for a in range(dimH):
        if check_blockade_violation_fast(a) == False:
            s_list[idx] = a
            idx = idx + 1
    s_array = s_list[:idx]
    return s_array

def create_H_PXP_blockade_sector(Delta, L, s_list):
    # H = Omega / 2 * Sx + Delta / 2 * Sz
    D = len(s_list)
    H = np.zeros((D, D), dtype = np.float64)
    for idx in range(D):
        a = s_list[idx]
        
        # Delta terms
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[idx,idx] = H[idx,idx] + Delta / 2 * (1 - 2*aj) 
        
        # XP terms
        j = 0
        aj = get_bit_j_fast(a, j)
        ajp1 = get_bit_j_fast(a, j+1)
        if ajp1 == 0:
            b = flip_bit_fast(a, j)
            idx1 = find_state(b, s_list)
            H[idx1, idx] = H[idx1, idx] + 1 / 2

        # PX terms
        j = L - 1
        aj = get_bit_j_fast(a, j)
        ajm1 = get_bit_j_fast(a, j-1)
        if ajm1 == 0:
            b = flip_bit_fast(a, j)
            idx1 = find_state(b, s_list)
            H[idx1, idx] = H[idx1, idx] + 1 / 2
            
        # PXP terms
        for j in range(1, L-1):
            aj = get_bit_j_fast(a, j)
            ajp1 = get_bit_j_fast(a, j+1)
            ajm1 = get_bit_j_fast(a, j-1)
            if ajp1 == 0 and ajm1 == 0:
                b = flip_bit_fast(a, j)
                idx1 = find_state(b, s_list)
                H[idx1, idx] = H[idx1, idx] + 1 / 2
         
    return H

def create_H_FSS_blockade_sector(Delta, Vnnn, L, s_list):
    # H = Omega / 2 * PXP + Delta / 2 * Z + Vnnn * Z_i * Z_i+2
    # Omega = 1
    D = len(s_list)
    H = np.zeros((D, D), dtype = np.float64)
    for idx in range(D):
        a = s_list[idx]
        
        # Delta terms
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            H[idx,idx] = H[idx,idx] + (Delta / 2 - Vnnn / 4) * (1 - 2*aj) 
        
        # Vnnn terms
        for j in range(L-2):
            aj = get_bit_j_fast(a,j)
            aj2 = get_bit_j_fast(a, j+2)
            H[idx,idx] = H[idx,idx] + Vnnn / 4 * (1 - 2*aj) * (1 - 2*aj2) 
        
        # XP terms
        j = 0
        aj = get_bit_j_fast(a, j)
        ajp1 = get_bit_j_fast(a, j+1)
        if ajp1 == 0:
            b = flip_bit_fast(a, j)
            idx1 = find_state(b, s_list)
            H[idx1, idx] = H[idx1, idx] + 1 / 2

        # PX terms
        j = L - 1
        aj = get_bit_j_fast(a, j)
        ajm1 = get_bit_j_fast(a, j-1)
        if ajm1 == 0:
            b = flip_bit_fast(a, j)
            idx1 = find_state(b, s_list)
            H[idx1, idx] = H[idx1, idx] + 1 / 2
            
        # PXP terms
        for j in range(1, L-1):
            aj = get_bit_j_fast(a, j)
            ajp1 = get_bit_j_fast(a, j+1)
            ajm1 = get_bit_j_fast(a, j-1)
            if ajp1 == 0 and ajm1 == 0:
                b = flip_bit_fast(a, j)
                idx1 = find_state(b, s_list)
                H[idx1, idx] = H[idx1, idx] + 1 / 2  
    return H

def create_H_FSS_blockade_sector_sparse(Delta, Vnnn, L, s_list):
    # H = Omega / 2 * PXP + Delta / 2 * Z + Vnnn * Z_i * Z_i+2
    # Omega = 1
    D = len(s_list)
    row  = np.zeros(D * L * 10, dtype = np.uint32)
    col  = np.zeros(D * L * 10, dtype = np.uint32)
    data = np.zeros(D * L * 10, dtype = np.float64)
    c = 0
    for idx in range(D):
        a = s_list[idx]
        
        # Delta terms
        for j in range(L):
            aj = get_bit_j_fast(a,j)
            row[c] = idx
            col[c] = idx
            data[c] = (Delta / 2 - Vnnn / 4) * (1 - 2*aj)
            c = c+1
    
        # Vnnn terms
        for j in range(L-2):
            aj = get_bit_j_fast(a,j)
            aj2 = get_bit_j_fast(a, j+2)
            row[c] = idx
            col[c] = idx
            data[c] =  Vnnn / 4 * (1 - 2*aj) * (1 - 2*aj2) 
            c = c+1
   
        # XP terms
        j = 0
        aj = get_bit_j_fast(a, j)
        ajp1 = get_bit_j_fast(a, j+1)
        if ajp1 == 0:
            b = flip_bit_fast(a, j)
            idx1 = find_state(b, s_list)
            row[c] = idx1
            col[c] = idx
            data[c] = 1 / 2
            c = c+1

        # PX terms
        j = L - 1
        aj = get_bit_j_fast(a, j)
        ajm1 = get_bit_j_fast(a, j-1)
        if ajm1 == 0:
            b = flip_bit_fast(a, j)
            idx1 = find_state(b, s_list)
            row[c] = idx1
            col[c] = idx
            data[c] = 1 / 2
            c = c+1
            
        # PXP terms
        for j in range(1, L-1):
            aj = get_bit_j_fast(a, j)
            ajp1 = get_bit_j_fast(a, j+1)
            ajm1 = get_bit_j_fast(a, j-1)
            if ajp1 == 0 and ajm1 == 0:
                b = flip_bit_fast(a, j)
                idx1 = find_state(b, s_list)
                row[c] = idx1
                col[c] = idx
                data[c] = 1 / 2
                c = c+1  
    H = coo_matrix((data[:c+1], (row[:c+1], col[:c+1])), shape = (D, D), dtype = np.float64)
    Hcsr = H.tocsr()
    return Hcsr