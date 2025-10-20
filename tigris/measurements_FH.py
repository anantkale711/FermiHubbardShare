import numpy as np
import scipy
import tigris.cython_files.hop_op_FH as hop_op_FH



def create_SzSz_func_sites(hop_op_list, vecs, tags, site_i, site_j):
    dimH = np.shape(hop_op_list[0][0][0])[0]
    SzSz_op =  np.zeros(dimH)
    # SziSzj = (n_upi-n_dni)*(n_upj-n_dnj) = n_upi n_upj + n_dni n_dnj - n_upi n_dnj - n_upj n_upi 
    ij_list = [(site_i, site_j)]
    for i,j in ij_list:
        SzSz_op += hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal()
        SzSz_op += hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal()
        SzSz_op += -hop_op_list[0][i][i].diagonal()*hop_op_list[1][j][j].diagonal()
        SzSz_op += -hop_op_list[1][i][i].diagonal()*hop_op_list[0][j][j].diagonal()
    SzSz_op /= len(ij_list)
    def apply_SzSz(state):
        result = SzSz_op * state
        return result
    return apply_SzSz

def create_doublon_func(hop_op_list, vecs, tags):
    dimH = np.shape(hop_op_list[0][0][0])[0]
    doublon_op = np.zeros(dimH)
    N_sites = len(hop_op_list[0])
    for i in range(N_sites):
        doublon_op += hop_op_list[0][i][i].diagonal()*hop_op_list[1][i][i].diagonal()
    doublon_op /= N_sites
    def apply_doublon(state):
        result = doublon_op * state
        return result
    return apply_doublon


def create_Sz_func_i(hop_op_list, vecs, tags, site_i):
    sz_op = hop_op_list[0][site_i][site_i].diagonal()-hop_op_list[1][site_i][site_i].diagonal()
    def apply_sz(state):
        result = sz_op * state
        return result
    return apply_sz
def create_density_func_i(hop_op_list, vecs, tags, site_i):
    n_op = hop_op_list[0][site_i][site_i].diagonal()+hop_op_list[1][site_i][site_i].diagonal()
    def apply_n(state):
        result = n_op * state
        return result
    return apply_n
def create_doublon_func_i(hop_op_list, vecs, tags, site_i):
    d_op = hop_op_list[0][site_i][site_i].diagonal()*hop_op_list[1][site_i][site_i].diagonal()
    def apply_d(state):
        result = d_op * state
        return result
    return apply_d

#====================================================
# Measurements avg over many bonds
#====================================================

def create_SzSz_func(hop_op_list, vecs, tags, bonds_list):
    dimH = np.shape(hop_op_list[0][0][0])[0]
    SzSz_op =  np.zeros(dimH)
    # SziSzj = (n_upi-n_dni)*(n_upj-n_dnj) = n_upi n_upj + n_dni n_dnj - n_upi n_dnj - n_upj n_upi 
    for i,j in bonds_list:
        SzSz_op += hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal()
        SzSz_op += hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal()
        SzSz_op += -hop_op_list[0][i][i].diagonal()*hop_op_list[1][j][j].diagonal()
        SzSz_op += -hop_op_list[1][i][i].diagonal()*hop_op_list[0][j][j].diagonal()
    SzSz_op /= len(bonds_list)
    def apply_SzSz(state):
        result = SzSz_op * state
        return result
    return apply_SzSz


def create_SxSx_func(hop_op_list, vecs, tags, bonds_list):
    dimH = np.shape(hop_op_list[0][0][0])[0]
    for site_i,site_j in bonds_list:
        if hop_op_list[0][site_i][site_j] == []:
            if site_i < site_j:
                i,j = site_i, site_j
            else:
                i,j = site_j, site_i
            for sigma in range(2):
                hop_op_list[sigma][i][j] = hop_op_FH.create_fermionic_hopping_operator_ij(i, j, vecs, tags)
                hop_op_list[sigma][j][i] = hop_op_list[sigma][i][j].transpose()
                    
    def apply_SxSx(state):
        final_result = np.zeros(dimH)
        for site_i,site_j in bonds_list:
            if site_j != site_i:
                result =  -hop_op_list[0][site_i][site_j]@hop_op_list[1][site_j][site_i]@state 
                result +=  -hop_op_list[0][site_j][site_i]@hop_op_list[1][site_i][site_j]@state
            else:
                result =  (hop_op_list[0][site_i][site_i].diagonal()+hop_op_list[1][site_i][site_i].diagonal())*state 
                result +=  -2*hop_op_list[0][site_i][site_i].diagonal()*hop_op_list[1][site_i][site_i].diagonal()*state 
            final_result += result
        # final_result.shape = dimH
        return final_result / len(bonds_list)
    return apply_SxSx


def create_nn_func(hop_op_list, vecs, tags, bonds_list):
    dimH = np.shape(hop_op_list[0][0][0])[0]
    nn_op =  np.zeros(dimH)
    for i,j in bonds_list:
        nn_op += hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal()
        nn_op += hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal()
        nn_op += hop_op_list[0][i][i].diagonal()*hop_op_list[1][j][j].diagonal()
        nn_op += hop_op_list[1][i][i].diagonal()*hop_op_list[0][j][j].diagonal()
    nn_op /= len(bonds_list)
    def apply_nn(state):
        result = nn_op * state
        return result
    return apply_nn


