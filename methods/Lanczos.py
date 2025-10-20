import numpy as np
import scipy.linalg
import time


def diagonalize_tridiagonal(a_list, n_list):
    return scipy.linalg.eigh_tridiagonal(a_list, n_list[1:-1])
    

# def Lanczos_diagonalize(H_func, init_state, max_iter = 1000, n_eigvals = 6, tol =1e-10, verbose = False, check_convergence=False):
#     eigvals_list = []
#     a_list = []
#     n_list = [None]
    
#     phi0  = init_state
#     phi1  = np.zeros(len(phi0))
#     phi2  = np.zeros(len(phi0))
    
#     H_func(init_state, out = phi1)
#     a0    = np.dot(phi0, phi1)  # all elements are real so no need to do vdot
    
#     #phi1 = phi1 - a0 * phi0   #implemented inplace
#     np.subtract(phi1, a0 * phi0, out = phi1)
#     n1    = np.sqrt(np.dot(phi1, phi1))
#     phi1 /= n1
    
#     a_list.append(a0)
#     n_list.append(n1)
    
#     #print(np.dot(phi1, phi1))
    
#     for i in range(1, max_iter):
#         H_func(phi1, out = phi2)
#         ai = np.dot(phi1, phi2)
#         ni = n_list[i]
#         #phi2 = phi2 - ai * phi1  - ni * phi0  
#         np.subtract(phi2, ai * phi1, out = phi2)
#         np.subtract(phi2, ni * phi0, out = phi2)
#         ni1 = np.sqrt(np.dot(phi2, phi2))
#         phi2 /= ni1
        
        
#         #print(np.dot(phi2, phi2), ', ', ai, ', ', ni1)
        
#         a_list.append(ai)
#         n_list.append(ni1)
        
#         # these should all just be pointers so no copying happening
#         temp = phi0
#         phi0 = phi1
#         phi1 = phi2
#         phi2 = temp
        
#         if (check_convergence and (i >= 50 and i%10 == 0)):
#             eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
#             eigvals_list.append(eigvals[:])
#             #print(eigvals_list[-1])
#             # if (i < n_eigvals+2): 
#             #     continue
#             if len(eigvals_list) > 2: # check for convergence
#                 if np.max(np.abs(eigvals_list[-1][:n_eigvals] - eigvals_list[-2][:n_eigvals])) < tol:
#                     return eigvals_list[-1], eigvecs[:,:], a_list, n_list 
    
#     #print('Reached max iterations (may not have converged eigvals!)')
#     eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
#     eigvals_list.append(eigvals[:])
#     return eigvals_list[-1], eigvecs[:,:], a_list, n_list


def reconstruct_Lanczos_vector(H_func_mat, init_state, vecs, a_list, n_list, out_list = None):
    if len(vecs.shape)==1:
        vecs = vecs.reshape(-1,1)
    Nv = vecs.shape[1]
    if out_list is None:
        psi_out_list = [np.zeros_like(init_state) for j in range(Nv)]
    else: 
        for out in out_list:
            out*= 0
        psi_out_list = out_list
    
    phi0  = init_state
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)
    
    for j in range(Nv):
        psi_out_list[j] += phi0 * vecs[0,j]
    
    H_func_mat(init_state, out = phi1)
    #phi1 = phi1 - a0 * phi0   #implemented inplace
    np.subtract(phi1, a_list[0] * phi0, out = phi1)
    phi1 /= n_list[1]
    
    for j in range(Nv):
        psi_out_list[j] += phi1 * vecs[1,j]
        
    for i in range(1, len(a_list)-1):
        H_func_mat(phi1, out = phi2)
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

def overlap_state_matrices(psi_m1, psi_m2):
    return np.sum(np.multiply(psi_m1, psi_m2))


def Lanczos_dynamical_response(H_func_mat, init_state, gs, Op_list, max_iter = 100):
    ''' 
    Assuming you already have the ground state and want to compute 
    C_AB(w) = FT[<gs|B(t)A|gs>]
    Performs Lanczos procedure with A|gs> (should be given as init_state)
    And for every lanczos basis state |phi> computes <gs|B|phi> for B in Op_list
    Assumes Operators are real (it actually computes and returns <phi|B|gs>)
    '''
    eigvals_list = []
    a_list = []
    n_list = [None]
    
    if not isinstance(Op_list, list):
        Op_list = [Op_list]
    
    Op_psi_list = [Op(gs) for Op in Op_list]
    overlaps_list = []
    
    phi0  = init_state
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)
    
    overlaps_list.append([overlap_state_matrices(phi0, Op_psi) for Op_psi in Op_psi_list])
    
    H_func_mat(init_state, out = phi1)
    a0    = overlap_state_matrices(phi0, phi1)  # all elements are real so no need to do vdot
    
    #phi1 = phi1 - a0 * phi0   #implemented inplace
    np.subtract(phi1, a0 * phi0, out = phi1)
    n1    = np.sqrt(overlap_state_matrices(phi1, phi1))
    phi1 /= n1
    
    
    a_list.append(a0)
    n_list.append(n1)
    
    #print(np.dot(phi1, phi1))
    
    for i in range(1, max_iter):
        overlaps_list.append([overlap_state_matrices(phi1, Op_psi) for Op_psi in Op_psi_list])
    
        H_func_mat(phi1, out = phi2)
        ai = overlap_state_matrices(phi1, phi2)
        ni = n_list[i]
        #phi2 = phi2 - ai * phi1  - ni * phi0  
        np.subtract(phi2, ai * phi1, out = phi2)
        np.subtract(phi2, ni * phi0, out = phi2)
        ni1 = np.sqrt(overlap_state_matrices(phi2, phi2))
        phi2 /= ni1
        
        
        #print(np.dot(phi2, phi2), ', ', ai, ', ', ni1)
        
        a_list.append(ai)
        n_list.append(ni1)
        
        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp
    
    eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
    eigvals_list.append(eigvals[:])
    return eigvals_list[-1], eigvecs[:,:], overlaps_list, a_list, n_list


def Lanczos_finite_temp(H_func, init_state, Op_list, n_eigvals = 2, max_iter = 100, tol=1e-10, check_convergence=False):
    ''' H_func should take a state_matrix instead of state vector'''
    eigvals_list = []
    a_list = []
    n_list = [None]
    
    if not isinstance(Op_list, list):
        Op_list = [Op_list]
    
    Op_psi_list = [Op(init_state) for Op in Op_list]
    overlaps_list = []
    
    phi0  = init_state
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)
    
    overlaps_list.append([overlap_state_matrices(phi0, Op_psi) for Op_psi in Op_psi_list])
    
    H_func(init_state, out = phi1)
    a0    = overlap_state_matrices(phi0, phi1)  # all elements are real so no need to do vdot
    
    #phi1 = phi1 - a0 * phi0   #implemented inplace
    np.subtract(phi1, a0 * phi0, out = phi1)
    n1    = np.sqrt(overlap_state_matrices(phi1, phi1))
    phi1 /= n1
    
    
    a_list.append(a0)
    n_list.append(n1)
    
    #print(np.dot(phi1, phi1))
    
    for i in range(1, max_iter):
        overlaps_list.append([overlap_state_matrices(phi1, Op_psi) for Op_psi in Op_psi_list])
    
        H_func(phi1, out = phi2)
        ai = overlap_state_matrices(phi1, phi2)
        ni = n_list[i]
        #phi2 = phi2 - ai * phi1  - ni * phi0  
        np.subtract(phi2, ai * phi1, out = phi2)
        np.subtract(phi2, ni * phi0, out = phi2)
        ni1 = np.sqrt(overlap_state_matrices(phi2, phi2))
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
        if (check_convergence and (i >= 50 and i%10 == 0)):
            eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
            eigvals_list.append(eigvals[:])
            #print(eigvals_list[-1])
            # if (i < n_eigvals+2): 
            #     continue
            if len(eigvals_list) > 2: # check for convergence
                if np.max(np.abs(eigvals_list[-1][:n_eigvals] - eigvals_list[-2][:n_eigvals])) < tol:
                    return eigvals_list[-1], eigvecs[:,:], overlaps_list, a_list, n_list 
    
    #print('Reached max iterations (may not have converged eigvals!)')
    eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
    eigvals_list.append(eigvals[:])
    return eigvals_list[-1], eigvecs[:,:], overlaps_list, a_list, n_list


def Lanczos_low_temp(H_func, init_state, Op_list, n_eigvals = 2, max_iter = 100, tol=1e-10):
    eigvals_list = []
    a_list = []
    n_list = [None]
    
    if not isinstance(Op_list, list):
        Op_list = [Op_list]
    
    # Run normal lanczos iteration once all the way through
    phi0  = np.copy(init_state)
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)
    
    H_func(phi0, out = phi1)
    a0    = overlap_state_matrices(phi0, phi1)  # all elements are real so no need to do vdot
    
    #phi1 = phi1 - a0 * phi0   #implemented inplace
    np.subtract(phi1, a0 * phi0, out = phi1)
    n1    = np.sqrt(overlap_state_matrices(phi1, phi1))
    phi1 /= n1
    
    a_list.append(a0)
    n_list.append(n1)
    
    for i in range(1, max_iter):
        H_func(phi1, out = phi2)
        ai = overlap_state_matrices(phi1, phi2)
        ni = n_list[i]
        #phi2 = phi2 - ai * phi1  - ni * phi0  
        np.subtract(phi2, ai * phi1, out = phi2)
        np.subtract(phi2, ni * phi0, out = phi2)
        ni1 = np.sqrt(overlap_state_matrices(phi2, phi2))
        phi2 /= ni1
        
        a_list.append(ai)
        n_list.append(ni1)
        
        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp
        
    n_iter = i+1
    print("Finished normal lanczos decomposition, ", n_iter)
    
    # Run the nested Lanczos to compute matrix elements
    overlaps_list = np.zeros((max_iter, max_iter, len(Op_list)))
    
    phi0  = np.copy(init_state)
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)
    for i in range(0, n_iter):
        # print(i)
        _phi0  = np.copy(init_state)
        _phi1  = np.zeros_like(phi0)
        _phi2  = np.zeros_like(_phi0)
        
        for j in range(0, i+1):
            overlaps_list[i,j] = np.array([overlap_state_matrices(phi1, Op(_phi1)) for Op in Op_list])

            if j==0:
                H_func(_phi0, out = _phi1)
                _a0 = a_list[0]
                _n1 = n_list[1]

                ## phi1 <== phi1 - a0 * phi0   #implemented inplace
                np.subtract(_phi1, _a0 * _phi0, out = _phi1)
                _phi1 /= _n1
                continue
        
            H_func(_phi1, out = _phi2)
            _ai = a_list[j]
            _ni = n_list[j]
            ## phi2 <== phi2 - ai * phi1  - ni * phi0  
            np.subtract(_phi2, _ai * _phi1, out = _phi2)
            np.subtract(_phi2, _ni * _phi0, out = _phi2)
            _ni1 = n_list[j+1]
            _phi2 /= _ni1
            
            # these should all just be pointers so no copying happening
            _temp = _phi0
            _phi0 = _phi1
            _phi1 = _phi2
            _phi2 = _temp
        
        if i==0:
            H_func(phi0, out = phi1)
            a0 = a_list[0]
            n1 = n_list[1]
            ## phi1 <== phi1 - a0 * phi0   #implemented inplace
            np.subtract(phi1, a0 * phi0, out = phi1)
            phi1 /= n1
            continue
    
        H_func(phi1, out = phi2)
        ai = a_list[i]
        ni = n_list[i]
        ## phi2 <== phi2 - ai * phi1  - ni * phi0  
        np.subtract(phi2, ai * phi1, out = phi2)
        np.subtract(phi2, ni * phi0, out = phi2)
        ni1 = n_list[i+1]
        phi2 /= ni1
        
        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp

    for i in range(0, n_iter):
        for j in range(i+1, n_iter):
            overlaps_list[i,j] = overlaps_list[j,i]  # these are all real so no complex conjugate needed
    
    # # transpose to get the first axis as the list of operators
    # overlaps_list = overlaps_list.moveaxis(overlaps_list, 2, 0)

    eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
    return eigvals, eigvecs, overlaps_list, a_list, n_list


def _Lanczos_iterate_compute(H_func, psi_next, psi, psi_prev, idx, ni=0):
    ''' Helper function that performs one lanczos iteration and returns ai,ni1'''
    if idx==0:
        ## phi1 = 1/ni1 * (H|phi0> -a0|phi0>)
        H_func(psi_prev, out = psi)
        a0 = overlap_state_matrices(psi, psi_prev)
        
        #phi1 = phi1 - a0 * phi0   #implemented inplace
        np.subtract(psi, a0 * psi_prev, out = psi)
        n1    = np.sqrt(overlap_state_matrices(psi, psi))
        psi /= n1
        return a0, n1
    else:
        ## phi2 = 1/ni1 * (H|phi1> -ai|phi1> -ni|phi0>)
        H_func(psi, out = psi_next)
        ai = overlap_state_matrices(psi, psi_next)
        ## phi2 <== phi2 - ai * phi1  - ni * phi0  
        np.subtract(psi_next, ai * psi, out = psi_next)
        np.subtract(psi_next, ni * psi_prev, out = psi_next)
        ni1 = np.sqrt(overlap_state_matrices(psi_next, psi_next))
        psi_next /= ni1     
        return ai, ni1


def _Lanczos_iterate(H_func, psi_next, psi, psi_prev, a_list, n_list, idx):
    ''' Helper function that performs one lanczos iteration (given a_list, n_list)'''
    if idx==0:
        ## phi1 = 1/ni1 * (H|phi0> -a0|phi0>)
        H_func(psi_prev, out = psi)
        a0 = a_list[0]
        n1 = n_list[1]
        ## phi1 <== phi1 - a0 * phi0   #implemented inplace
        np.subtract(psi, a0 * psi_prev, out = psi)
        psi /= n1
        return
    else:
        ## phi2 = 1/ni1 * (H|phi1> -ai|phi1> -ni|phi0>)
        H_func(psi, out = psi_next)
        ai = a_list[idx]
        ni = n_list[idx]
        ## phi2 <== phi2 - ai * phi1  - ni * phi0  
        np.subtract(psi_next, ai * psi, out = psi_next)
        np.subtract(psi_next, ni * psi_prev, out = psi_next)
        ni1 = n_list[idx+1]
        psi_next /= ni1            
        return


def Lanczos_low_temp_block(H_func_mat, init_state, Op_list, n_block=10, max_iter = 100):
    assert max_iter%n_block == 0
    a_list = []
    n_list = [0]

    if not isinstance(Op_list, list):
        Op_list = [Op_list]
    
    # Run normal lanczos iteration once all the way through
    phi0  = np.copy(init_state)
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)

    a0,n1 = _Lanczos_iterate_compute(H_func_mat,phi2, phi1, phi0, 0)
    a_list.append(a0)
    n_list.append(n1)
    
    for i in range(1, max_iter):
        ai,ni1 = _Lanczos_iterate_compute(H_func_mat,phi2, phi1, phi0, i, n_list[i])
        a_list.append(ai)
        n_list.append(ni1)
        
        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp
        
    n_iter = max_iter
    print("Finished normal lanczos decomposition, ", n_iter)
    
    # Run the nested Lanczos to compute matrix elements
    overlaps_list = np.zeros((max_iter, max_iter, len(Op_list)))
    
    states_block = [[] for i in range(n_block)]
    
    for bi in range(0, n_iter//n_block):
        phi0  = np.copy(init_state)
        phi1  = np.zeros_like(phi0)
        phi2  = np.zeros_like(phi0)
        _Lanczos_iterate(H_func_mat, phi2, phi1, phi0, a_list, n_list, 0)

        if bi==0: 
            states_block[0] = np.copy(phi0)
            
        # lanczos iterate until we obtain n_block basis states
        for i in range(1, (bi+1)*n_block):
            if i >=bi*n_block:
                states_block[i-bi*n_block] = np.copy(phi1)
            _Lanczos_iterate(H_func_mat, phi2, phi1, phi0, a_list, n_list, i)
            # these should all just be pointers so no copying happening
            temp = phi0
            phi0 = phi1
            phi1 = phi2
            phi2 = temp

        # compute matrix elements for <bi|A|bj> for j<=i
        for i in range(n_block):
            for opi, Op in enumerate(Op_list):
                temp = Op(states_block[i])
                for j in range(i+1):
                    overlaps_list[bi*n_block+i,bi*n_block+j, opi] = overlap_state_matrices(temp, states_block[j])

        
        # lanczos iterate again starting from 0 until the current block
        phi0  = np.copy(init_state)
        phi1  = np.zeros_like(phi0)
        phi2  = np.zeros_like(phi0)
        _Lanczos_iterate(H_func_mat, phi2, phi1, phi0, a_list, n_list, 0)
        if bi>0:
            for opi, Op in enumerate(Op_list):
                temp = Op(phi0)
                for i in range(n_block):
                    overlaps_list[bi*n_block+i,0, opi] = overlap_state_matrices(temp, states_block[i])
        for i in range(1, bi*n_block):
            # for each basis state compute <phi|A|bi> for bi in state block
            for opi, Op in enumerate(Op_list):
                temp = Op(phi1)
                for j in range(n_block):
                    overlaps_list[bi*n_block+j,i, opi] = overlap_state_matrices(temp, states_block[j])
            _Lanczos_iterate(H_func_mat, phi2, phi1, phi0, a_list, n_list, i)
            # these should all just be pointers so no copying happening
            temp = phi0
            phi0 = phi1
            phi1 = phi2
            phi2 = temp

    # populate the upper right triangle of overlaps
    for i in range(0, n_iter):
        for j in range(i+1, n_iter):
            overlaps_list[i,j] = overlaps_list[j,i]  # these are all real so no complex conjugate needed


    eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
    return eigvals, eigvecs, overlaps_list, a_list, n_list
    
def Lanczos_diagonalize_mat(H_func_mat, init_state, max_iter = 100):
    n_iter = max_iter

    a_list = []
    n_list = [0]

    # Run normal lanczos iteration once all the way through
    phi0  = np.copy(init_state)
    phi1  = np.zeros(phi0.shape)
    phi2  = np.zeros(phi0.shape)
    a0,n1 = _Lanczos_iterate_compute(H_func_mat,phi2, phi1, phi0, 0)
    a_list.append(a0)
    n_list.append(n1)
    # print(a0,n1)
    for i in range(1, max_iter):
        ai,ni1 = _Lanczos_iterate_compute(H_func_mat,phi2, phi1, phi0, i, n_list[i]) 
        a_list.append(ai)
        n_list.append(ni1)
        # print(ai,ni1)
        
        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp
    print("Finished normal lanczos decomposition, ", n_iter)
    eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
    return eigvals, eigvecs, a_list, n_list
    # return a_list, n_list

def Lanczos_dynamical_response_FTLM(H_func_mat, init_state, OpA, OpB, n_block=10, max_iter = 100):
    ''' Compute the matrix elements needed to evaluate C(t) = <B(t)A>
    The simplest case is the autocorrelation function when B = A^\dagger '''
    assert max_iter%n_block == 0
    n_iter = max_iter

    a_list = []
    n_list = [0]
    
    a1_list = []
    n1_list = [0]
    perturbed_state = OpA(init_state)
    state_norm = np.sqrt(overlap_state_matrices(perturbed_state, perturbed_state))
    perturbed_state /= state_norm

    tic = time.time()
    # Run normal lanczos iteration once all the way through
    phi0  = np.copy(init_state)
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)
    a0,n1 = _Lanczos_iterate_compute(H_func_mat,phi2, phi1, phi0, 0)
    a_list.append(a0)
    n_list.append(n1)
    for i in range(1, max_iter):
        ai,ni1 = _Lanczos_iterate_compute(H_func_mat,phi2, phi1, phi0, i, n_list[i])
        a_list.append(ai)
        n_list.append(ni1)
        
        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp
    toc= time.time()
    print(f"Finished normal lanczos decomposition in {toc-tic:.3f} seconds")
    
    tic = time.time()
    # Run normal lanczos iteration again with Op|psi> all the way through
    phi0  = np.copy(perturbed_state)
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)
    a0,n1 = _Lanczos_iterate_compute(H_func_mat,phi2, phi1, phi0, 0)
    a1_list.append(a0)
    n1_list.append(n1)
    for i in range(1, max_iter):
        ai,ni1 = _Lanczos_iterate_compute(H_func_mat,phi2, phi1, phi0, i, n1_list[i])
        a1_list.append(ai)
        n1_list.append(ni1)
        
        # these should all just be pointers so no copying happening
        temp = phi0
        phi0 = phi1
        phi1 = phi2
        phi2 = temp

    toc= time.time()
    print(f"Finished normal lanczos decomposition of Op|psi> in {toc-tic:.3f} seconds")

    # Run the nested Lanczos to compute matrix elements
    overlaps_list = np.zeros((max_iter, max_iter))
    
    tic = time.time()
    # allocate huge amounts of memory, but doesn't have to be all contiguous since its a list of arrays
    states_block = [np.zeros_like(phi0) for i in range(n_block)] 
    toc= time.time()
    # print(f"Allocated memory in {toc-tic:.4f} seconds")

    phi0  = np.copy(init_state)
    phi1  = np.zeros_like(phi0)
    phi2  = np.zeros_like(phi0)
    _Lanczos_iterate(H_func_mat, phi2, phi1, phi0, a_list, n_list, 0)

    for bi in range(0, n_iter//n_block):
        tic = time.time()

        if bi==0: 
            states_block[0] *= 0
            states_block[0] += phi0
            
        # lanczos iterate until we obtain n_block basis states
        for i in range(bi*n_block+1, (bi+1)*n_block):
            if i >=bi*n_block:
                states_block[i-bi*n_block] *= 0
                states_block[i-bi*n_block] += phi1
            _Lanczos_iterate(H_func_mat, phi2, phi1, phi0, a_list, n_list, i)
            # these should all just be pointers so no copying happening
            temp = phi0
            phi0 = phi1
            phi1 = phi2
            phi2 = temp
        toc= time.time()
        # print(f"Made {n_block} block states in {toc-tic:.4f} seconds")
        
        tic = time.time()
        # lanczos iterate again starting from 0 until the current block
        # start with perturbed state A|psi>
        phi0_  = np.copy(perturbed_state)
        phi1_  = np.zeros_like(phi0_)
        phi2_  = np.zeros_like(phi0_)
        _Lanczos_iterate(H_func_mat, phi2_, phi1_, phi0_, a1_list, n1_list, 0)
        temp_ = OpB(phi0_)
        for j in range(n_block):
            overlaps_list[bi*n_block+j,0] = overlap_state_matrices(temp_, states_block[j])
        for i in range(1, (bi+1)*n_block):
            # for each basis state compute <phi|B|bi> for bi in state block
            temp_ = OpB(phi1_)
            if i >= bi*n_block:
                for j in range(i-bi*n_block, n_block):
                    overlaps_list[bi*n_block+j,i] = overlap_state_matrices(temp_, states_block[j])
            _Lanczos_iterate(H_func_mat, phi2_, phi1_, phi0_, a1_list, n1_list, i)
            # these should all just be pointers so no copying happening
            temp_ = phi0_
            phi0_ = phi1_
            phi1_ = phi2_
            phi2_ = temp_
        toc= time.time()
        # print(f"Computed overlaps in {toc-tic:.4f} seconds")

    # populate the upper right triangle of overlaps
    for i in range(0, n_iter):
        for j in range(i+1, n_iter):
            overlaps_list[i,j] = overlaps_list[j,i]  # these are all real so no complex conjugate needed

    eigvals, eigvecs = diagonalize_tridiagonal(a_list, n_list)
    eigvals1, eigvecs1 = diagonalize_tridiagonal(a1_list, n1_list)
    return eigvals, eigvecs, eigvals1, eigvecs1, overlaps_list, state_norm, a_list, n_list, a1_list, n1_list
    
