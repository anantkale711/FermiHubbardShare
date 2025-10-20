import numpy as np
import scipy.linalg
import leo.hamiltonians3 as hamiltonians3


def diagonalize_tridiagonal_complex(a_list, n_list):
#     D = len(a_list)
#     a_band = np.zeros((2,D), dtype=complex)
#     a_band[0] = a_list
#     a_band[1,:-1] = n_list[1:-1]
#     return scipy.linalg.eig_banded(a_band, lower=True)
    return scipy.linalg.eigh_tridiagonal(a_list, n_list[1:-1])
    

def Lanczos_diagonalize_complex(H_func, init_state, n_eigvals = 6, max_iter = 1000, tol =1e-10, verbose = False):
    eigvals_list = []
    a_list = []
    n_list = [None]
    
    phi0  = np.copy(init_state).astype(complex)
    phi1  = np.zeros(len(phi0), dtype=complex)
    phi2  = np.zeros(len(phi0), dtype=complex)
    
    H_func(init_state, out = phi1)
    a0    = np.vdot(phi0, phi1).real  # elements might be complex so we do need to do vdot
    # earlier all elements were real so no need to do vdot
    
    #phi1 = phi1 - a0 * phi0   #implemented inplace
    np.subtract(phi1, a0 * phi0, out = phi1)
    n1    = np.sqrt(np.vdot(phi1, phi1).real)
    phi1 /= n1
    
    a_list.append(a0)
    n_list.append(n1)
    
    #print(np.dot(phi1, phi1))
    
    for i in range(1, max_iter):
        H_func(phi1, out = phi2)
        ai = np.vdot(phi1, phi2).real
        ni = n_list[i]
        #phi2 = phi2 - ai * phi1  - ni * phi0  
        np.subtract(phi2, ai * phi1, out = phi2)
        np.subtract(phi2, ni * phi0, out = phi2)
        ni1 = np.sqrt(np.vdot(phi2, phi2).real)
        phi2 /= ni1
        
        
        #print(np.dot(phi2, phi2), ', ', ai, ', ', ni1)
        
        a_list.append(ai)
        n_list.append(ni1)
        
        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp
        
#         print(a_list)
#         print(n_list)
        eigvals, eigvecs = diagonalize_tridiagonal_complex(a_list, n_list)
        eigvals_list.append(eigvals[:n_eigvals])
        #print(eigvals_list[-1])
        if (i < n_eigvals+2): 
            continue
        if np.max(np.abs(eigvals_list[-1] - eigvals_list[-2])) < tol:
            print(i)
            if verbose: 
                return eigvals_list, eigvecs[:,:n_eigvals], a_list, n_list
            return eigvals_list[i], eigvecs[:,:n_eigvals], a_list, n_list
    print('Reached max iterations, may not have converged eigvals!')
    if verbose: 
        return eigvals_list, eigvecs[:,:n_eigvals], a_list, n_list
    return eigvals_list[-1], eigvecs[:,:n_eigvals], a_list, n_list


def reconstruct_Lanczos_vector_complex(H_func, init_state, vecs, a_list, n_list, out_list = None):
    if len(vecs.shape)==1:
        vecs = vecs.reshape(-1,1)
    Nv = vecs.shape[1]
    if out_list is None:
        psi_out_list = [np.zeros(len(init_state), dtype=complex) for j in range(Nv)]
    else: 
        for out in out_list:
            out*= 0
        psi_out_list = out_list
    
    phi0  = np.copy(init_state).astype(complex)
    phi1  = np.zeros(len(phi0), dtype=complex)
    phi2  = np.zeros(len(phi0), dtype=complex)
    
    for j in range(Nv):
        psi_out_list[j] += phi0 * vecs[0,j]
    
    H_func(init_state, out = phi1)
    #phi1 = phi1 - a0 * phi0   #implemented inplace
    np.subtract(phi1, a_list[0] * phi0, out = phi1)
    phi1 /= n_list[1]
    
    for j in range(Nv):
        psi_out_list[j] += phi1 * vecs[1,j]
        
    for i in range(1, len(a_list)-1):
        H_func(phi1, out = phi2)
        #phi2 = phi2 - ai * phi1  - ni * phi0  
        np.subtract(phi2, a_list[i] * phi1, out = phi2)
        np.subtract(phi2, n_list[i] * phi0, out = phi2)
        phi2 /= n_list[i+1]
        
        for j in range(Nv):
            psi_out_list[j] += phi2 * vecs[i+1,j]

        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp
    if Nv == 1:
        return psi_out_list[0]
    else:
        return psi_out_list
