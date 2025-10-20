import leo.hamiltonians3 as hamiltonians3
import methods.Lanczos as Lanczos
from leo.hamiltonian_helper import *
from pathlib import Path
import gc
import leo.cython_files.hop_op_FH as hop_op_FH
import os
import scipy
import time


def run_particular_symmetry_sector(N_sites, N_up, N_down, dimH, U, modulation,  connectivity_list, conx, cony, direc, N_states=100, seed_start=1000):
    
    # N_states = lanczos expansion order
    N_cutoff = 500 # dense ED to sparse ED cutoff

    tic0 = time.time()
    states_up    = hamiltonians3.generate_N_sector_states_FH(N_sites,N_up)
    states_down  = hamiltonians3.generate_N_sector_states_FH(N_sites,N_down)
    tic1 = time.time()
    print(f'Made states lists in {tic1-tic0} seconds')
    states_lists = [states_up, states_down]
    hop_op_list  = hamiltonians3.create_fermionic_hopping_operators_from_conn_list(states_lists, connectivity_list)
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    Hk_ux, Hk_dx = hamiltonians3.create_H_kin_FH(1.0, hop_op_list, conx)
    Hk_uy, Hk_dy = hamiltonians3.create_H_kin_FH(1.0, hop_op_list, cony)
    Hi = hamiltonians3.create_H_int_FH(U, hop_op_list)
    tic2 = time.time()
    print(f'Made Hamiltonians in {tic2-tic1} seconds')

    tic3 = time.time()
    print(f'Made Operators in {tic3-tic2} seconds')
    
    seed_val = seed_start
    print('Starting seed: {}'.format(seed_val))
    np.random.seed(seed_val)
        
    filename = os.path.join(direc, "Nup_%d_Ndown_%d_U_%.1f_seed_%d.npz"%(N_up, N_down,U, seed_val))

    if dimH > N_cutoff:
        print(f'Starting lanczos for ({N_sites},{N_up},{N_down}): {dimH}')
        
        Hku = Hk_ux + Hk_uy
        Hkd = Hk_dx + Hk_dy
        H_func = create_H_func_new_mat(Hku, Hkd, Hi)

        tic = time.time()
        psi = np.random.uniform(-1,1, size = dimH)
        psi /= np.sqrt(np.dot(psi, psi))
        psi.shape=(dimH_up, dimH_down)
        psi_copy = np.copy(psi)
        evals, evecs, a_list, n_list = Lanczos.Lanczos_diagonalize_mat(H_func, psi, N_states)
        toc = time.time()
        print(f'Found ground state in {toc-tic} seconds, E_gnd = {evals[0]:.4f}')

        tic = time.time()
        gs = Lanczos.reconstruct_Lanczos_vector(H_func, psi_copy, evecs[:,0], a_list, n_list)
        gs.shape=(dimH_up, dimH_down)
        toc = time.time()
        print(f'Reconstructed state vector in {toc-tic} seconds')

        tic = time.time()
        # apply perturbation
        if modulation=='nematic':
            Hru =  Hk_uy - Hk_ux
            Hrd =  Hk_dy - Hk_dx
            gsp =  apply_operator_on_state_helper(Hru, 0, gs)
            gsp += apply_operator_on_state_helper(Hrd, 1, gs)
        else:
            Hku =  Hk_uy + Hk_ux
            Hkd =  Hk_dy + Hk_dx
            gsp =  apply_operator_on_state_helper(Hku, 0, gs)
            gsp += apply_operator_on_state_helper(Hkd, 1, gs)
        # apply perturbation
        # gsp = Op_list[0](gs)
        # gsp.shape=dimH
        # gsp_norm = np.sqrt(np.dot(gsp, gsp))
        gsp_norm = np.sqrt(np.sum(np.multiply(gsp, gsp)))
        gsp /= gsp_norm

        gsp_copy = np.copy(gsp)
        evalsp, evecsp, a_listp, n_listp = Lanczos.Lanczos_diagonalize_mat(H_func, gsp, N_states)
        toc = time.time()
        print(f'Second Lanczos procedure in {toc-tic} seconds, E_gnd = {evalsp[0]:.4f}')

        savez_dict = dict()
        savez_dict['evals'] = evals
        savez_dict['evecs'] = evecs
        savez_dict['evalsp'] = evalsp
        savez_dict['evecsp'] = evecsp
        savez_dict['a_listp'] = a_listp
        savez_dict['n_listp'] = [0]+n_listp[1:]
        savez_dict['gsp_norm'] = gsp_norm

        with open(filename, 'wb') as f:
            np.savez(f, **savez_dict)
        print('Saved file for seed: {}'.format(seed_val))

    else:
        print(f'Do dense ED for ({N_sites},{N_up},{N_down}): {dimH}')
        
    toc0 = time.time()
    print(f'Dynamical response in {toc0-tic0} seconds')
    gc.collect()
    return toc0-tic0   
