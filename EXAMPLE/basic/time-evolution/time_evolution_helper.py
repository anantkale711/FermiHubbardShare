import leo.hamiltonians3 as hamiltonians3
import methods.Lanczos as Lanczos
from leo.hamiltonian_helper import *
from pathlib import Path
import gc
import leo.cython_files.hop_op_FH as hop_op_FH
import os
import scipy
import time



def run_particular_symmetry_sector(N_sites, U, N_up, N_down, measurements, ramp_duration, dimH, connectivity_list, conx, cony, direc, imb_i, imb_f, N_states=100, seed_start=1000):
    
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
    Hk_ux, Hk_dx = hamiltonians3.create_H_kin_FH(1.0, hop_op_list, conx)
    Hk_uy, Hk_dy = hamiltonians3.create_H_kin_FH(1.0, hop_op_list, cony)
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
    imbalance = imb_i
    Hku = Hk_ux * (1-imbalance) + Hk_uy * (1+imbalance)
    Hkd = Hk_dx * (1-imbalance) + Hk_dy * (1+imbalance)
    H_func = create_H_func_new_mat(Hku, Hkd, Hi)

    if dimH > N_cutoff:
        print(f'Starting lanczos for ({N_sites},{N_up},{N_down}): {dimH}')
        seed_val = seed_start
        print('Starting seed: {}'.format(seed_val))
        np.random.seed(seed_val)
        
        tic = time.time()
        psi = np.random.uniform(-1,1, size = dimH)
        psi /= np.sqrt(np.dot(psi, psi))
        psi.shape=(dimH_up, dimH_down)
        psi_copy = np.copy(psi)

        evals, evecs, a_list, n_list = Lanczos.Lanczos_diagonalize_mat(H_func, psi, N_states)
        toc = time.time()
        print(f'Found ground state in {toc-tic} seconds, E_gnd = {evals[0]:.4f}')

        full_eigvecs = Lanczos.reconstruct_Lanczos_vector(H_func, psi_copy, evecs[:,0], a_list, n_list)
        toc2 = time.time()
        print(f'Reconstructed state vector in {toc2-toc} seconds')

        filename = os.path.join(direc, "Nup_%d_Ndown_%d_U_%.1f_seed_%d_imbalance_%.2f_%.2f.npz"%(N_up, N_down,U, seed_val, imb_i, imb_f))
        Path(filename[:-4]).mkdir(parents=True, exist_ok=True)

        gs = full_eigvecs
        psi = np.copy(gs)
        imbalance = imb_f
        Hku = Hk_ux * (1-imbalance) + Hk_uy * (1+imbalance)
        Hkd = Hk_dx * (1-imbalance) + Hk_dy * (1+imbalance)
        dt = 0.05
        N_steps = int(ramp_duration/dt)
        times = np.linspace(0,N_steps*dt, N_steps+1, endpoint=True)
        results = sim_ramp_psi(ramp_duration, dt, psi, Hku, Hkd, Hi, Op_list, qnames, filename)
        
        savez_dict = dict()
        for q_i in range(len(Op_list)):
            savez_dict[qnames[q_i]] = results[q_i]
        savez_dict['overlap'] = results[-1]
        savez_dict['norm'] = results[-2]
        savez_dict['evals'] = evals
        savez_dict['times'] = times
        
        with open(filename, 'wb') as f:
            np.savez(f, **savez_dict)
            print('Saved file for seed: {}'.format(seed_val))

    else:
        print('Do dense ED')
        # print(f'Starting dense ED for ({N_sites},{N_up},{N_down}): {dimH}')
        # H = np.zeros((dimH, dimH))
        # Iden_u = np.identity(np.shape(hop_op_list[0][0][0])[0])
        # Iden_d = np.identity(np.shape(hop_op_list[1][0][0])[0])
        # H += np.kron(Hk_u.toarray(), Iden_d) + np.kron(Iden_u, Hk_d.toarray())
        # H += U*np.diag(Hi.reshape(dimH))
        # eigvals, eigvecs = scipy.linalg.eigh(H)
        
    toc = time.time()
    gc.collect()
    return toc-tic   


def sim_ramp_psi(ramp_duration, dt, gs, H_kin_up, H_kin_down, H_int, Op_list, qnames, filename):
    
    # Choose time-step
    N_steps = int(ramp_duration / dt)
    times = np.linspace(0,N_steps*dt, N_steps+1, endpoint=True)
    # Trotter
    # Perform ramp time-evolution
    tic = time.time()
    # U_kin_up = scipy.linalg.expm(-1j * H_kin_up.toarray() * dt)
    # U_kin_down = scipy.linalg.expm(-1j * H_kin_down.toarray() * dt)
    U_int_2 = np.exp(-1j * H_int * dt/2)
    toc = time.time()
    print(f'Made unitary operators in {toc-tic} seconds')
    dimH_up, dimH_down = H_int.shape
    tic = time.time()
    gs_copy = np.copy(gs)
    gs_copy.shape=(dimH_up, dimH_down)
    psi = np.array(gs, dtype = complex)
    psi.shape=(dimH_up, dimH_down)
    temp_arr = np.zeros(np.shape(gs), dtype = complex)
    res = np.zeros((2+len(Op_list), N_steps+1))
    tic = time.time()
    for j in range(len(Op_list)):
        res[j,0] = np.sum(np.multiply(Op_list[j](psi).conj(), psi)).real
    res[-1,0] = np.abs(np.sum(np.multiply(psi.conj(), gs_copy)))**2
    res[-2,0] = np.abs(np.sum(np.multiply(psi.conj(), psi)))**2
    for i in range(1,N_steps+1):
        # print(i)
        np.multiply(U_int_2, psi, out = psi)
        psi_new = scipy.sparse.linalg.expm_multiply(-1j*H_kin_up*dt, psi)
        psi = psi_new.transpose()
        psi_new = scipy.sparse.linalg.expm_multiply(-1j*H_kin_down*dt, psi)
        psi = psi_new.transpose()
        np.multiply(U_int_2, psi, out = psi)
        norm_psi = np.sqrt(np.sum(np.multiply(psi.conj(), psi)).real)
        psi /= norm_psi
        for j in range(len(Op_list)):
            res[j,i] = np.sum(np.multiply(Op_list[j](psi).conj(), psi)).real
        res[-1,i] = np.abs(np.sum(np.multiply(psi.conj(), gs_copy)))**2
        res[-2,i] =  norm_psi**2
        savez_dict = dict()
        for q_i in range(len(Op_list)):
            savez_dict[qnames[q_i]] = res[q_i,:i+1]
        savez_dict['overlap'] = res[-1,:i+1]
        savez_dict['norm'] = res[-2,:i+1]
        savez_dict['times'] = times[:i+1]
        new_fname = os.path.join(filename[:-4], f"{i}.npz")
        with open(new_fname, 'wb') as f:
            np.savez(f, **savez_dict)
        print('saved', i)

    toc = time.time()
    print(f'Time evolved in {toc-tic} seconds')
    return res