import leo.hamiltonians2 as hamiltonians2
import leo.hamiltonians3 as hamiltonians3
import leo.Lanczos as Lanczos
# import leo.cython_files.Lanczos as Lanczos
from leo.hamiltonian_helper import *
import gc
import leo.cython_files.hop_op_FH as hop_op_FH
import os
import scipy
import time



def run_particular_symmetry_sector(N_sites, U, N_up, N_down, measurements, N_seeds, dimH, connectivity_list, direc, N_states=100, seed_start=1000):
    
    # N_states = lanczos expansion order
    N_cutoff = 500 # dense ED to sparse ED cutoff

    
    tic = time.time()
    states_up    = hamiltonians3.generate_N_sector_states_FH(N_sites,N_up)
    states_down  = hamiltonians3.generate_N_sector_states_FH(N_sites,N_down)
    tic1 = time.time()
    print(f'Made states lists in {tic1-tic} seconds')
    states_lists = [states_up, states_down]
    hop_op_list  = hamiltonians3.create_fermionic_hopping_operators_from_conn_list(states_lists, connectivity_list)
    Hk_u, Hk_d = hamiltonians3.create_H_kin_FH(1.0, hop_op_list, connectivity_list)
    Hi = hamiltonians3.create_H_int_FH(1.0, hop_op_list)
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

    H_func_mat = create_H_func_new_mat(Hk_u, Hk_d, U*Hi)

    if dimH > N_cutoff:
        print(f'Starting lanczos for ({N_sites},{N_up},{N_down}): {dimH}')
        for r_i in range(N_seeds):
            seed_val = r_i + seed_start
            print('Starting seed: {}'.format(seed_val))
            np.random.seed(seed_val)
            
            psi = np.random.uniform(-1,1, size = dimH)
            psi /= np.sqrt(np.dot(psi, psi))
            psi.shape = Hi.shape
            evals, evecs, overlaps_list, a_list, n_list = Lanczos.Lanczos_low_temp(H_func_mat, psi, Op_list, 4, N_states, 1e-10)

            # mLanczos = len(overlaps_list)
            # print(mLanczos)
            # mpad = max(0, N_states-mLanczos)
            overlaps_array = np.array(overlaps_list)
            for q_i in range(len(Op_list)):
                # temp_arr = np.zeros((N_states, N_states))
                # temp_arr[:mLanczos,:mLanczos] = overlaps_array[:,:,q_i]
                # results[q_i][r_i] = temp_arr
                results[q_i][r_i] = overlaps_array[:,:,q_i]
            
            new_evals = np.array(evals)
            # new_evals = np.array(list(evals)+list(np.ones(mpad)*1e6))
            # new_evals = np.array(list(evals[-1])+list(np.ones(mpad)*1e6))
            result_evals[r_i] = new_evals
            new_evecs = evecs
            # new_evecs = np.zeros((N_states, N_states))
            # new_evecs[:mLanczos, :mLanczos] = evecs
            # result_evecs[r_i] = new_evecs
            
            savez_dict = dict()
            for q_i in range(len(Op_list)):
                savez_dict[qnames[q_i]] = results[q_i][r_i]
            savez_dict['evecs'] = new_evecs
            savez_dict['evals'] = new_evals
            
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
        H += U*np.diag(Hi.reshape(dimH))
        eigvals, eigvecs = scipy.linalg.eigh(H)
        evals = eigvals
        overlaps_list = []
        for psi_i in range(len(eigvals)):
            psi = eigvecs[:,psi_i]
            overlaps_list.append([np.dot(psi, Op(psi)) for Op in Op_list])
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




def run_particular_symmetry_sector_block(N_sites, U, N_up, N_down, measurements, N_seeds, dimH, connectivity_list, direc, N_states=100, N_block=10, seed_start=1000):
    
    # N_states = lanczos expansion order
    N_cutoff = 500 # dense ED to sparse ED cutoff

    
    tic = time.time()
    states_up    = hamiltonians3.generate_N_sector_states_FH(N_sites,N_up)
    states_down  = hamiltonians3.generate_N_sector_states_FH(N_sites,N_down)
    tic1 = time.time()
    print(f'Made states lists in {tic1-tic} seconds')
    states_lists = [states_up, states_down]
    hop_op_list  = hamiltonians3.create_fermionic_hopping_operators_from_conn_list(states_lists, connectivity_list)
    Hk_u, Hk_d = hamiltonians3.create_H_kin_FH(1.0, hop_op_list, connectivity_list)
    Hi = hamiltonians3.create_H_int_FH(1.0, hop_op_list)
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

    H_func_mat = create_H_func_new_mat(Hk_u, Hk_d, U*Hi)

    if dimH > N_cutoff:
        print(f'Starting lanczos for ({N_sites},{N_up},{N_down}): {dimH}')
        for r_i in range(N_seeds):
            seed_val = r_i + seed_start
            print('Starting seed: {}'.format(seed_val))
            np.random.seed(seed_val)
            
            psi = np.random.uniform(-1,1, size = dimH)
            psi /= np.sqrt(np.dot(psi, psi))
            psi.shape=Hi.shape

            evals, evecs, overlaps_list, a_list, n_list = Lanczos.Lanczos_low_temp_block(H_func_mat, psi, Op_list, N_block, N_states)

            overlaps_array = np.array(overlaps_list)
            for q_i in range(len(Op_list)):
                results[q_i][r_i] = overlaps_array[:,:,q_i]
            
            new_evals = np.array(evals)
            result_evals[r_i] = new_evals
            new_evecs = evecs
            
            savez_dict = dict()
            for q_i in range(len(Op_list)):
                savez_dict[qnames[q_i]] = results[q_i][r_i]
            savez_dict['evecs'] = new_evecs
            savez_dict['evals'] = new_evals
            
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
        H += U*np.diag(Hi.reshape(dimH))
        eigvals, eigvecs = scipy.linalg.eigh(H)
        evals = eigvals
        overlaps_list = []
        for psi_i in range(len(eigvals)):
            psi = eigvecs[:,psi_i]
            overlaps_list.append([np.dot(psi, Op(psi)) for Op in Op_list])
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