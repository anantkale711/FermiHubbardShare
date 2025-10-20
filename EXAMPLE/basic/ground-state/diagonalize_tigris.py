import tigris.hamiltonians2 as hamiltonians2
from tigris import measurements_FH
import methods.Lanczos as Lanczos
from pathlib import Path
import gc
import tigris.cython_files.hop_op_FH as hop_op_FH
import os
import scipy
import time


import sys
import numpy as np
from scipy.special import comb
from methods.geometry import *
import random

show_plots = True
# show_plots = False

if show_plots:
    import matplotlib.pyplot as plt


d2r = np.pi/180.
e1 = np.array([1,0])
e2 = np.array([0,1])

ce1 = np.array([1,0])
ce2 = np.array([0,1])
# ce3 = np.array([1,1])

# Tvec12 = np.array([-2,-3])
# Tvec34 = np.array([4,-2])
# Tvec12 = np.array([3,0])
# Tvec34 = np.array([0,3])

Tvec12 = np.array([8,0])
Tvec34 = np.array([0,1])
Tvecs = [Tvec12, -Tvec12, Tvec34, -Tvec34]

lines = generate_line_functions(Tvecs, e1, e2, flip=False)
count, sites, sdict = generate_lattice_sites(e1, e2, lines)
print(count)
N_sites= count
    
def make_geometry(boundary):
    cvecs = [ce1, -ce1, ce2, -ce2]

    if boundary == "PBC":
        connectivity_list = generate_connectivity_list(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    else:
        connectivity_list = generate_connectivity_list_OBC(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    
    folder_name = "Square_%d_(%d,%d)_(%d,%d)_%s"%(N_sites, Tvecs[0][0], Tvecs[0][1], Tvecs[2][0], Tvecs[2][1], boundary)
    
    geom_dict = {'e1':e1, 'e2':e2, 'cvecs':cvecs, 'Tvecs': Tvecs, 'lines':lines, 'sites':sites, 'sdict': sdict, 'bc':boundary}

    if show_plots:
        print(connectivity_list)
        # print('Here')
        fig, ax = plt.subplots()
        if boundary == "PBC":
            ax.arrow(0,0, *(Tvecs[0][0]*e1 + Tvecs[0][1]*e2), width=0.08, zorder=1.)
            ax.arrow(0,0, *(Tvecs[2][0]*e1 + Tvecs[2][1]*e2), width=0.08, zorder=1.)

        for i, (xi,yi) in enumerate(sites):
            x,y = xi*e1 + yi*e2
            for s1 in connectivity_list[i]:
                x1, y1 = sites[s1][0]*e1 + sites[s1][1]*e2
                ax.arrow(x,y,x1-x, y1-y,color='black', zorder=1.0)

        for i, (xi,yi) in enumerate(sites):
            x,y = xi*e1 + yi*e2
            s = plt.Circle((x,y), 0.4, fc='lightskyblue', ec='darkblue', alpha=0.9, zorder=1000.)
            ax.add_patch(s)
            ax.text(x,y, "%d"%i, ha='center', va='center', size=10, zorder=10001.)
        plt.xlim(-1,5)
        plt.ylim(-1,5)
        # plt.xlim(-0.5,4)
        # plt.ylim(-4.5,1.5)

        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()
    return connectivity_list, folder_name, geom_dict



def make_measurements_list(geom_dict):
    cvecs = geom_dict['cvecs']
    bc = geom_dict['bc']
    # ce1 = cvecs[0]
    # ce2 = cvecs[2]
    # cvecsNN = cvecs
    # cvecsNNNc = [ce1+ce2, ce1-ce2, -ce1-ce2, ce1+ce2]
    # cvecsNNNs = [2*ce for ce in cvecs]
    # cvecsNNNNc = [ce1+2*ce2, ce2+2*ce1, ce1-2*ce2, ce2-2*ce1, -ce1-2*ce2, -ce2-2*ce1, -ce1+2*ce2, -ce2+2*ce1]

    if bc == "PBC":
        NN1_bonds = generate_bonds_list_PBC(geom_dict['sites'], geom_dict['sdict'], geom_dict['lines'], geom_dict['Tvecs'], 
                                        cvecs[:2], geom_dict['e1'], geom_dict['e2'])
        
        NN2_bonds = generate_bonds_list_PBC(geom_dict['sites'], geom_dict['sdict'], geom_dict['lines'], geom_dict['Tvecs'], 
                                        cvecs[2:], geom_dict['e1'], geom_dict['e2'])
    else:
        NN1_bonds = generate_bonds_list_OBC(geom_dict['sites'], geom_dict['sdict'], geom_dict['lines'], geom_dict['Tvecs'], 
                                        cvecs[:2], geom_dict['e1'], geom_dict['e2'])
        
        NN2_bonds = generate_bonds_list_OBC(geom_dict['sites'], geom_dict['sdict'], geom_dict['lines'], geom_dict['Tvecs'], 
                                        cvecs[2:], geom_dict['e1'], geom_dict['e2'])
        
    quantities = [

              (measurements_FH.create_SzSz_func_sites,(0,1,)),
              (measurements_FH.create_SzSz_func_sites,(0,2,)),
              (measurements_FH.create_SzSz_func_sites,(0,3,)),
              (measurements_FH.create_SzSz_func_sites,(0,4,)),
              (measurements_FH.create_SzSz_func_sites,(0,5,)),
              (measurements_FH.create_SzSz_func_sites,(0,6,)),
              (measurements_FH.create_SzSz_func_sites,(0,7,)),

            #   (measurements_FH.create_SzSz_func,(NN1_bonds,)),
            #   (measurements_FH.create_SzSz_func,(NN2_bonds,)),   
            #   (measurements_FH.create_pp_func,(NN1_bonds,)),
            #   (measurements_FH.create_pp_func,(NN2_bonds,)),   
            #   (measurements_FH.create_nn_func,(NN1_bonds,)),
            #   (measurements_FH.create_nn_func,(NN2_bonds,)),   
    ]
    return quantities

def create_H_func(H_kin, H_int):
    def apply_H(state, out=None):
        if out is None:
            result = np.zeros(np.shape(state))
        else:
            out *= 0
            result = out
        result += (H_kin+H_int)@state
        return result
    return apply_H

def make_hamiltonian_and_diagonalize(N_sites, N_up, N_down, U, connectivity_list, Operator_list, N_states=100):
    '''
    Constructs hamiltonians and diagonalizes to find ground state
    provide connectivity_list which defines the geometry of the system
    provide Operator_list to measure things about the ground state 
    (operators should be function names and arguments which will be generated after hamiltonians are constructed
    returns a function that take a state as input)
    '''
    N_cutoff = 500 # dense ED to sparse ED cutoff

    tic0 = time.time()
    vecs, tags = hamiltonians2.generate_state_vecs_and_tags_FH(N_sites, N_up, N_down)
    tic1 = time.time()
    print(f'Made states lists in {tic1-tic0} seconds')
    
    hop_op_list  = hamiltonians2.create_fermionic_hopping_operators_from_conn_list(vecs, tags, connectivity_list)
    H_kin = hamiltonians2.create_H_FH_kin_from_conn_list(1, hop_op_list, connectivity_list)
    H_int = hamiltonians2.create_H_FH_int(U, hop_op_list)
    dimH = np.shape(H_int)[0]
    tic2 = time.time()
    print(f'Made Hamiltonians in {tic2-tic1} seconds')

    Op_list = [[] for q in Operator_list]
    qnames = []
    for q_i, (func,args) in enumerate(Operator_list):
        Op_list[q_i] = func(hop_op_list, vecs, tags, *args)
        qnames.append(str(func.__name__)+str(args))
    tic3 = time.time()
    print(f'Made Operators in {tic3-tic2} seconds')

    if dimH > N_cutoff:
        print(f'Starting lanczos for ({N_sites},{N_up},{N_down}): {dimH}')
        
        H_func = create_H_func(H_kin, H_int)

        tic = time.time()
        psi = np.random.uniform(-1,1, size = dimH)
        psi /= np.sqrt(np.dot(psi, psi))
        psi_copy = np.copy(psi)
        evals, evecs, a_list, n_list = Lanczos.Lanczos_diagonalize_mat(H_func, psi, N_states)
        toc = time.time()
        print(f'Found ground state in {toc-tic} seconds, E_gnd = {evals[0]:.4f}')

        tic = time.time()
        gs = Lanczos.reconstruct_Lanczos_vector(H_func, psi_copy, evecs[:,0], a_list, n_list)
        toc = time.time()
        print(f'Reconstructed state vector in {toc-tic} seconds')


    else:
        print(f'Starting dense ED for ({N_sites},{N_up},{N_down}): {dimH}')
        H = H_kin.toarray() + H_int.toarray()
        eigvals, eigvecs = scipy.linalg.eigh(H)
        gs = eigvecs[:,0]
    
    measurements = [np.sum(np.multiply(gs, Op(gs))) for Op in Op_list]
    toc0 = time.time()
    print(f'Full time= {toc0-tic0} seconds')
    gc.collect()
    return qnames, measurements

if __name__== '__main__':
    bc = "OBC"
    connectivity_list, folder_name, geom_dict = make_geometry(bc)
    
    N_sites = len(connectivity_list)
    
    N_up=4
    N_down=4

    measurement_operators = make_measurements_list(geom_dict)

    U = -4
    N_states = 250

    seed_start = random.randint(1e6,1e7)
    print("Seed start integer = ", seed_start)

    np.random.seed(seed_start)
    results = make_hamiltonian_and_diagonalize(N_sites, N_up, N_down, U, connectivity_list, measurement_operators, N_states)
    print(results[0])
    print(results[1])
    if show_plots:
        plt.figure()
        plt.plot(range(len(results[1])), results[1])
        plt.grid()
        plt.show()