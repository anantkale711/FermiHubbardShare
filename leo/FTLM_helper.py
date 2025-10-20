import leo.hamiltonians3 as hamiltonians3
import methods.Lanczos as Lanczos
from leo.hamiltonian_helper import *
import gc
import leo.cython_files.hop_op_FH as hop_op_FH
import os
import scipy
import time



def run_particular_symmetry_sector(N_sites, U, N_up, N_down, measurements, N_seeds, connectivity_list, direc, N_states=100, seed_start=1000):
    
    # N_states = lanczos expansion order
    N_cutoff = 500 # dense ED to sparse ED cutoff

    
    tic = time.time()
    states_up    = hamiltonians3.generate_N_sector_states_FH(N_sites,N_up)
    states_down  = hamiltonians3.generate_N_sector_states_FH(N_sites,N_down)
    tic1 = time.time()
    print(f'Made states lists in {tic1-tic} seconds')
    states_lists = [states_up, states_down]
    hop_op_list  = hamiltonians3.create_fermionic_hopping_operators_from_conn_list(states_lists, connectivity_list)
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    Hk_u, Hk_d = hamiltonians3.create_H_kin_FH(1.0, hop_op_list, connectivity_list)
    Hi = hamiltonians3.create_H_int_FH(U, hop_op_list)
    tic2 = time.time()
    print(f'Made Hamiltonians in {tic2-tic1} seconds')
    
    Op_list = [[] for q in measurements]
    qnames = []
    for q_i, (func,args) in enumerate(measurements):
        Op_list[q_i] = func(hop_op_list, states_lists, *args)
        qnames.append(str(func.__name__)+str(args))
    tic3 = time.time()
    print(f'Made Operators in {tic3-tic2} seconds')
    
    results = [[[] for r_i in range(N_seeds)] for q_i,q in enumerate(measurements)]
    result_evals = [[] for r_i in range(N_seeds)] 
    result_evecs = [[] for r_i in range(N_seeds)]

    H_func = create_H_func_new_mat(Hk_u, Hk_d, Hi)

    if dimH > N_cutoff:
        print(f'Starting lanczos for ({N_sites},{N_up},{N_down}): {dimH}')
        for r_i in range(N_seeds):
            seed_val = r_i + seed_start
            print('Starting seed: {}'.format(seed_val))
            np.random.seed(seed_val)
            
            psi = np.random.uniform(-1,1, size = dimH)
            psi /= np.sqrt(np.dot(psi, psi))
            psi.shape=(dimH_up, dimH_down)

            evals, evecs, overlaps_list, a_list, n_list = Lanczos.Lanczos_finite_temp(H_func, psi, Op_list, 4, N_states, 1e-10)
            overlaps_array = np.array(overlaps_list)
            for q_i in range(len(Op_list)):
                results[q_i][r_i] = np.array(overlaps_array[:, q_i])
                
            result_evals[r_i] = evals
            result_evecs[r_i] = evecs
            
            savez_dict = dict()
            for q_i in range(len(Op_list)):
                savez_dict[qnames[q_i]] = results[q_i][r_i]
            savez_dict['evecs'] = evecs
            savez_dict['evals'] = evals
            
            filename = os.path.join(direc, "Nup_%d_Ndown_%d_U_%.1f_seed_%d.npz"%(N_up, N_down,U, seed_val))
            with open(filename, 'wb') as f:
                np.savez(f, **savez_dict)
                print('Saved file for seed: {}'.format(seed_val))
            
            
    else:
        print(f'Starting dense ED for ({N_sites},{N_up},{N_down}): {dimH}')
        H = np.zeros((dimH, dimH))
        Iden_u = np.identity(np.shape(hop_op_list[0][0][0])[0])
        Iden_d = np.identity(np.shape(hop_op_list[1][0][0])[0])
        H += np.kron(Hk_u.toarray(), Iden_d) + np.kron(Iden_u, Hk_d.toarray())
        H += np.diag(Hi.reshape(dimH))
        eigvals, eigvecs = scipy.linalg.eigh(H)
        evals = eigvals
        overlaps_list = []
        for psi_i in range(len(eigvals)):
            psi = eigvecs[:,psi_i]
            psi.shape=(dimH_up, dimH_down)
            overlaps_list.append([np.sum(np.multiply(psi, Op(psi))) for Op in Op_list])
        overlaps_array = np.array(overlaps_list)
        
        r_i = 0
        seed_val = 0
        for q_i in range(len(Op_list)):
            results[q_i][r_i] = overlaps_array[:, q_i]
            
        result_evals[r_i] = evals
        
        savez_dict = dict()
        for q_i in range(len(Op_list)):
            savez_dict[qnames[q_i]] = results[q_i][r_i]
        savez_dict['evals'] = evals

        filename = os.path.join(direc,"Nup_%d_Ndown_%d_U_%.1f_seed_%d.npz"%(N_up, N_down, U, seed_val))
        with open(filename, 'wb') as f:
            np.savez(f, **savez_dict)
            print(f'Saved file for dense ED: ({N_sites},{N_up},{N_down}): {dimH}')
    toc = time.time()
    gc.collect()
    return toc-tic   