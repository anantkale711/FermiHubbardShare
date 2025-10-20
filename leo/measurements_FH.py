import numpy as np
import scipy
from leo.hamiltonian_helper import *
import leo.cython_files.hop_op_FH as hop_op_FH




def create_SzSz_func_sites(hop_op_list, states_lists, site_i, site_j):
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    SzSz_op =  np.zeros((dimH_up, dimH_down))
    Iden_u = np.ones(dimH_up) 
    Iden_d = np.ones(dimH_down) 
    temp  = np.zeros((dimH_up, dimH_down))
    # SziSzj = (n_upi-n_dni)*(n_upj-n_dnj) = n_upi n_upj + n_dni n_dnj - n_upi n_dnj - n_upj n_upi 
    ij_list = [(site_i, site_j)]
    for i,j in ij_list:
        np.outer(hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal(), Iden_d, out = temp)
        np.multiply(temp, 1/ len(ij_list), out=temp)
        SzSz_op += temp 
        np.outer(Iden_u, hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, 1/ len(ij_list), out=temp)
        SzSz_op += temp 
        np.outer(hop_op_list[0][i][i].diagonal(), hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, -1/ len(ij_list), out=temp)
        SzSz_op += temp 
        np.outer(hop_op_list[0][j][j].diagonal(),hop_op_list[1][i][i].diagonal(),  out = temp)
        np.multiply(temp, -1/ len(ij_list), out=temp)
        SzSz_op += temp 
    del temp
    def apply_SzSz(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = SzSz_op * state_matrix
        # result.shape=dimH
        return result
    return apply_SzSz

def create_Stot_func(hop_op_list, states_lists):
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    N_sites = len(hop_op_list[0])
    print(N_sites)
    for site_i in range(N_sites):
        for site_j in range(N_sites):
            if hop_op_list[0][site_i][site_j] == []:
                if site_i < site_j:
                    i,j = site_i, site_j
                elif site_i > site_j:
                    i,j = site_j, site_i
                else: 
                    continue
                for sigma in range(2):
                    hop_op_list[sigma][i][j] = hop_op_FH.create_fermionic_hopping_operator_ij(i, j, states_lists[sigma])
                    hop_op_list[sigma][j][i] = hop_op_list[sigma][i][j].transpose()
                    
    def apply_Stot(state):
        state_matrix = state.reshape(dimH_up, dimH_down)
        final_result = np.zeros((dimH_up, dimH_down))
        for site_i in range(N_sites):
            # print(site_i)
            ni_u = hop_op_list[0][site_i][site_i]
            ni_d = hop_op_list[1][site_i][site_i]
            for site_j in range(N_sites):
                nj_u = hop_op_list[0][site_j][site_j]
                nj_d = hop_op_list[1][site_j][site_j]
                if site_j != site_i:
                    result1 =  1/2*apply_operator_on_state_helper(ni_u, 0, state_matrix)
                    result1 += -1/2*apply_operator_on_state_helper(ni_d, 1, state_matrix)
                    result =  1/2*apply_operator_on_state_helper(nj_u, 0, result1)
                    result += -1/2*apply_operator_on_state_helper(nj_d, 1, result1)

                    result +=  -1/2*apply_operator_list_on_state([(hop_op_list[0][site_i][site_j], 0), 
                                                        (hop_op_list[1][site_j][site_i], 1)], state_matrix)
                    result +=  -1/2*apply_operator_list_on_state([(hop_op_list[0][site_j][site_i], 0), 
                                                        (hop_op_list[1][site_i][site_j], 1)], state_matrix)
                else:
                    # continue
                    Iden_u = scipy.sparse.identity(np.shape(hop_op_list[0][0][0])[0])
                    Iden_d = scipy.sparse.identity(np.shape(hop_op_list[1][0][0])[0])
                    hi_u = Iden_u - ni_u
                    hi_d = Iden_d - ni_d
                    result  = 3/4*apply_operator_list_on_state([(ni_u, 0), (hi_d, 1)], state_matrix)
                    result += 3/4*apply_operator_list_on_state([(ni_d, 1), (hi_u, 0)], state_matrix)
                    # print(site_i, site_j)
                final_result += result
        final_result.shape = dimH
        return final_result
    return apply_Stot



# def create_doublon_func(hop_op_list, states_lists):
#     dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
#     dimH = dimH_up*dimH_down
#     N_sites = len(hop_op_list[0])
#     def apply_doublon(state):
#         state_matrix = state.reshape(dimH_up, dimH_down)
#         result = np.zeros((dimH_up, dimH_down))
#         for i in range(N_sites):
#             result += apply_operator_list_on_state([(hop_op_list[0][i][i], 0), 
#                                                     (hop_op_list[1][i][i], 1)], 
#                                                     state_matrix)
#         result.shape = dimH
#         return result / N_sites
#     return apply_doublon


def create_doublon_func(hop_op_list, states_lists):
    dimH_up = np.shape(hop_op_list[0][0][0])[0]
    dimH_down = np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    doublon_op = np.zeros((dimH_up, dimH_down))
    temp  = np.zeros((dimH_up, dimH_down))
    N_sites = len(hop_op_list[0])
    for i in range(N_sites):
        np.outer(hop_op_list[0][i][i].diagonal(), hop_op_list[1][i][i].diagonal(), out = temp)
        np.multiply(temp, 1/N_sites, out=temp)
        doublon_op += temp 
    del temp
    def apply_doublon(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = doublon_op * state_matrix
        # result.shape=dimH
        return result
        
    return apply_doublon


# def create_singles_func(hop_op_list, states_lists):
#     dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
#     dimH = dimH_up*dimH_down
#     N_sites = len(hop_op_list[0])
#     Iden_u = scipy.sparse.identity(np.shape(hop_op_list[0][0][0])[0])
#     Iden_d = scipy.sparse.identity(np.shape(hop_op_list[1][0][0])[0])
#     def apply_singles(state):
#         state_matrix = state.reshape(dimH_up, dimH_down)
#         result = np.zeros((dimH_up, dimH_down))
#         for i in range(N_sites):
#             result += apply_operator_list_on_state([(hop_op_list[0][i][i], 0), 
#                                                     (Iden_d-hop_op_list[1][i][i], 1)], state_matrix)
#             result += apply_operator_list_on_state([(hop_op_list[1][i][i], 1), 
#                                                     (Iden_u-hop_op_list[0][i][i], 0)], state_matrix)
#         result.shape = dimH
#         return result/ N_sites
#     return apply_singles

def create_singles_func(hop_op_list, states_lists):
    dimH_up = np.shape(hop_op_list[0][0][0])[0]
    dimH_down = np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    singles_op = np.zeros((dimH_up, dimH_down))
    temp  = np.zeros((dimH_up, dimH_down))
    Iden_u = np.ones(dimH_up) 
    Iden_d = np.ones(dimH_down) 
    N_sites = len(hop_op_list[0])
    for i in range(N_sites):
        np.outer(hop_op_list[0][i][i].diagonal(), Iden_d - hop_op_list[1][i][i].diagonal(), out = temp)
        np.multiply(temp, 1/N_sites, out=temp)
        singles_op += temp 
        np.outer(Iden_u - hop_op_list[0][i][i].diagonal(), hop_op_list[1][i][i].diagonal(), out = temp)
        np.multiply(temp, 1/N_sites, out=temp)
        singles_op += temp 
    del temp
    def apply_singles(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = singles_op * state_matrix
        # result.shape=dimH
        return result
        
    return apply_singles

def create_Sz_func_i(hop_op_list, states_lists, site_i):
    dimH_up = np.shape(hop_op_list[0][0][0])[0]
    dimH_down = np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    sz_op = np.zeros((dimH_up, dimH_down))
    temp  = np.zeros((dimH_up, dimH_down))
    Iden_u = np.ones(dimH_up) 
    Iden_d = np.ones(dimH_down) 
    N_sites = len(hop_op_list[0])
    np.outer(hop_op_list[0][site_i][site_i].diagonal(), Iden_d, out = temp)
    sz_op += temp 
    np.outer(Iden_u, hop_op_list[1][site_i][site_i].diagonal(), out = temp)
    sz_op -= temp 
    del temp
    def apply_sz(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = sz_op * state_matrix
        # result.shape=dimH
        return result
    return apply_sz


def create_density_func_i(hop_op_list, states_lists, site_i):
    dimH_up = np.shape(hop_op_list[0][0][0])[0]
    dimH_down = np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    n_op = np.zeros((dimH_up, dimH_down))
    temp  = np.zeros((dimH_up, dimH_down))
    Iden_u = np.ones(dimH_up) 
    Iden_d = np.ones(dimH_down) 
    np.outer(hop_op_list[0][site_i][site_i].diagonal(), Iden_d, out = temp)
    n_op += temp 
    np.outer(Iden_u, hop_op_list[1][site_i][site_i].diagonal(), out = temp)
    n_op += temp 
    del temp
    def apply_n(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = n_op * state_matrix
        # result.shape=dimH
        return result
    return apply_n

def create_doublon_func_i(hop_op_list, states_lists, site_i):
    dimH_up = np.shape(hop_op_list[0][0][0])[0]
    dimH_down = np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    doublon_op = np.zeros((dimH_up, dimH_down))
    np.outer(hop_op_list[0][site_i][site_i].diagonal(), hop_op_list[1][site_i][site_i].diagonal(), out = doublon_op)
    def apply_doublon(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = doublon_op * state_matrix
        # result.shape=dimH
        return result
    return apply_doublon

#====================================================
# OBC measurements (avg over the entire lattice)
#====================================================

# def create_SzSz_func_OBC(hop_op_list, states_lists, sites, sdict, cvec_list):
#     dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
#     dimH = dimH_up*dimH_down
#     ij_list = []
#     for i,s in enumerate(sites):
#         for cvec in cvec_list:
#             s1 = s + cvec
#             if tuple(s1) in sdict.keys():
#                 j = sdict[tuple(s1)]
#                 ij_list.append((i,j))

#     def apply_SzSz(state):
#         state_matrix = state.reshape(dimH_up, dimH_down)
#         final_result = np.zeros((dimH_up, dimH_down))
#         for i,j in ij_list:
#             result =  apply_operator_on_state_helper(hop_op_list[0][i][i], 0, state_matrix)
#             result += -apply_operator_on_state_helper(hop_op_list[1][i][i], 1, state_matrix)
#             result2 =  apply_operator_on_state_helper(hop_op_list[0][j][j], 0, result)
#             result2 += -apply_operator_on_state_helper(hop_op_list[1][j][j], 1, result)
#             final_result += result2
#         final_result.shape = dimH
#         return final_result / len(ij_list)
#     return apply_SzSz

def create_SzSz_func_OBC(hop_op_list, states_lists, sites, sdict, cvec_list):
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    ij_list = []
    for i,s in enumerate(sites):
        for cvec in cvec_list:
            s1 = s + cvec
            if tuple(s1) in sdict.keys():
                j = sdict[tuple(s1)]
                ij_list.append((i,j))
    SzSz_op =  np.zeros((dimH_up, dimH_down))
    Iden_u = np.ones(dimH_up) 
    Iden_d = np.ones(dimH_down) 
    temp  = np.zeros((dimH_up, dimH_down))
    # SziSzj = (n_upi-n_dni)*(n_upj-n_dnj) = n_upi n_upj + n_dni n_dnj - n_upi n_dnj - n_upj n_upi 
    for i,j in ij_list:
        np.outer(hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal(), Iden_d, out = temp)
        np.multiply(temp, 1/ len(ij_list), out=temp)
        SzSz_op += temp 
        np.outer(Iden_u, hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, 1/ len(ij_list), out=temp)
        SzSz_op += temp 
        np.outer(hop_op_list[0][i][i].diagonal(), hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, -1/ len(ij_list), out=temp)
        SzSz_op += temp 
        np.outer(hop_op_list[0][j][j].diagonal(),hop_op_list[1][i][i].diagonal(),  out = temp)
        np.multiply(temp, -1/ len(ij_list), out=temp)
        SzSz_op += temp 
    del temp
    def apply_SzSz(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = SzSz_op * state_matrix
        # result.shape=dimH
        return result
    return apply_SzSz

def create_SxSx_func_OBC(hop_op_list, states_lists, sites, sdict, cvec_list):
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    ij_list = []
    for i,s in enumerate(sites):
        for cvec in cvec_list:
            s1 = s + cvec
            if tuple(s1) in sdict.keys():
                j = sdict[tuple(s1)]
                ij_list.append((i,j))
    
    for site_i,site_j in ij_list:
        if hop_op_list[0][site_i][site_j] == []:
            if site_i < site_j:
                i,j = site_i, site_j
            else:
                i,j = site_j, site_i
            for sigma in range(2):
                hop_op_list[sigma][i][j] = hop_op_FH.create_fermionic_hopping_operator_ij(i, j, states_lists[sigma])
                hop_op_list[sigma][j][i] = hop_op_list[sigma][i][j].transpose()
                    
    def apply_SxSx(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        final_result = np.zeros((dimH_up, dimH_down))
        for site_i,site_j in ij_list:
            if site_j != site_i:
                result =  -apply_operator_list_on_state([(hop_op_list[0][site_i][site_j], 0), 
                                                     (hop_op_list[1][site_j][site_i], 1)], state_matrix)
                result +=  -apply_operator_list_on_state([(hop_op_list[0][site_j][site_i], 0), 
                                                      (hop_op_list[1][site_i][site_j], 1)], state_matrix)
            else:
                result =  apply_operator_on_state_helper(hop_op_list[0][site_i][site_j], 0, state_matrix)
                result +=  apply_operator_on_state_helper(hop_op_list[1][site_i][site_j], 1, state_matrix)
            final_result += result
        # final_result.shape = dimH
        return final_result / len(ij_list)
    return apply_SxSx


def create_nn_func_OBC(hop_op_list, states_lists, sites, sdict, cvec_list):
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    ij_list = []
    for i,s in enumerate(sites):
        for cvec in cvec_list:
            s1 = s + cvec
            if tuple(s1) in sdict.keys():
                j = sdict[tuple(s1)]
                ij_list.append((i,j))

    def apply_nn(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        final_result = np.zeros((dimH_up, dimH_down))
        for i,j in ij_list:
            result =  apply_operator_on_state_helper(hop_op_list[0][i][i], 0, state_matrix)
            result += apply_operator_on_state_helper(hop_op_list[1][i][i], 1, state_matrix)
            result2 =  apply_operator_on_state_helper(hop_op_list[0][j][j], 0, result)
            result2 += apply_operator_on_state_helper(hop_op_list[1][j][j], 1, result)
            final_result += result2
        # final_result.shape = dimH
        return final_result / len(ij_list)
    return apply_nn

#====================================================
# PBC measurements (avg over the entire lattice)
#====================================================


# def create_SzSz_func_PBC(hop_op_list, states_lists, bonds_list):
#     dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
#     dimH = dimH_up*dimH_down
    
#     def apply_SzSz(state):
#         state_matrix = state.reshape(dimH_up, dimH_down)
#         final_result = np.zeros((dimH_up, dimH_down))
#         for i,j in bonds_list:
#             result =  apply_operator_on_state_helper(hop_op_list[0][i][i], 0, state_matrix)
#             result += -apply_operator_on_state_helper(hop_op_list[1][i][i], 1, state_matrix)
#             result2 =  apply_operator_on_state_helper(hop_op_list[0][j][j], 0, result)
#             result2 += -apply_operator_on_state_helper(hop_op_list[1][j][j], 1, result)
#             final_result += result2
#         final_result.shape = dimH
#         return final_result / len(bonds_list)
#     return apply_SzSz

def create_SzSz_func(hop_op_list, states_lists, bonds_list):
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    
    SzSz_op =  np.zeros((dimH_up, dimH_down))
    Iden_u = np.ones(dimH_up) 
    Iden_d = np.ones(dimH_down) 
    temp  = np.zeros((dimH_up, dimH_down))
    # SziSzj = (n_upi-n_dni)*(n_upj-n_dnj) = n_upi n_upj + n_dni n_dnj - n_upi n_dnj - n_upj n_upi 
    for i,j in bonds_list:
        np.outer(hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal(), Iden_d, out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        SzSz_op += temp 
        np.outer(Iden_u, hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        SzSz_op += temp 
        np.outer(hop_op_list[0][i][i].diagonal(), hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, -1/ len(bonds_list), out=temp)
        SzSz_op += temp 
        np.outer(hop_op_list[0][j][j].diagonal(),hop_op_list[1][i][i].diagonal(),  out = temp)
        np.multiply(temp, -1/ len(bonds_list), out=temp)
        SzSz_op += temp 
    del temp
    def apply_SzSz(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = SzSz_op * state_matrix
        # result.shape=dimH
        return result
    return apply_SzSz


def create_SxSx_func(hop_op_list, states_lists, bonds_list):
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    
    for site_i,site_j in bonds_list:
        if hop_op_list[0][site_i][site_j] == []:
            if site_i < site_j:
                i,j = site_i, site_j
            else:
                i,j = site_j, site_i
            for sigma in range(2):
                hop_op_list[sigma][i][j] = hop_op_FH.create_fermionic_hopping_operator_ij(i, j, states_lists[sigma])
                hop_op_list[sigma][j][i] = hop_op_list[sigma][i][j].transpose()
                    
    def apply_SxSx(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        final_result = np.zeros((dimH_up, dimH_down))
        for site_i,site_j in bonds_list:
            if site_j != site_i:
                result =  -apply_operator_list_on_state([(hop_op_list[0][site_i][site_j], 0), 
                                                     (hop_op_list[1][site_j][site_i], 1)], state_matrix)
                result +=  -apply_operator_list_on_state([(hop_op_list[0][site_j][site_i], 0), 
                                                      (hop_op_list[1][site_i][site_j], 1)], state_matrix)
            else:
                result =  apply_operator_on_state_helper(hop_op_list[0][site_i][site_j], 0, state_matrix)
                result +=  apply_operator_on_state_helper(hop_op_list[1][site_i][site_j], 1, state_matrix)
            final_result += result
        # final_result.shape = dimH
        return final_result / len(bonds_list)
    return apply_SxSx


# def create_nn_func_PBC(hop_op_list, states_lists, bonds_list):
#     dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
#     dimH = dimH_up*dimH_down

#     def apply_SzSz(state):
#         state_matrix = state.reshape(dimH_up, dimH_down)
#         final_result = np.zeros((dimH_up, dimH_down))
#         for i,j in bonds_list:
#             result =  apply_operator_on_state_helper(hop_op_list[0][i][i], 0, state_matrix)
#             result += apply_operator_on_state_helper(hop_op_list[1][i][i], 1, state_matrix)
#             result2 =  apply_operator_on_state_helper(hop_op_list[0][j][j], 0, result)
#             result2 += apply_operator_on_state_helper(hop_op_list[1][j][j], 1, result)
#             final_result += result2
#         final_result.shape = dimH
#         return final_result / len(bonds_list)
#     return apply_SzSz

def create_nn_func(hop_op_list, states_lists, bonds_list):
    dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    
    nn_op =  np.zeros((dimH_up, dimH_down))
    Iden_u = np.ones(dimH_up) 
    Iden_d = np.ones(dimH_down) 
    temp  = np.zeros((dimH_up, dimH_down))
    # ninj = (n_upi+n_dni)*(n_upj+n_dnj) = n_upi n_upj + n_dni n_dnj + n_upi n_dnj + n_upj n_upi 
    for i,j in bonds_list:
        np.outer(hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal(), Iden_d, out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        nn_op += temp 
        np.outer(Iden_u, hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        nn_op += temp 
        np.outer(hop_op_list[0][i][i].diagonal(), hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        nn_op += temp 
        np.outer(hop_op_list[0][j][j].diagonal(),hop_op_list[1][i][i].diagonal(),  out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        nn_op += temp 
    del temp
    def apply_nn(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = nn_op * state_matrix
        # result.shape=dimH
        return result
    return apply_nn

# def create_pp_func_PBC(hop_op_list, states_lists, bonds_list):
#     dimH_up, dimH_down = np.shape(hop_op_list[0][0][0])[0], np.shape(hop_op_list[1][0][0])[0]
#     dimH = dimH_up*dimH_down
#     Iden_u = scipy.sparse.identity(np.shape(hop_op_list[0][0][0])[0])
#     Iden_d = scipy.sparse.identity(np.shape(hop_op_list[1][0][0])[0])
#     def apply_pp(state):
#         state_matrix = state.reshape(dimH_up, dimH_down)
#         final_result = np.zeros((dimH_up, dimH_down))
#         for i,j in bonds_list:
#             result =  apply_operator_list_on_state([(hop_op_list[0][i][i], 0), 
#                                                     (Iden_d-hop_op_list[1][i][i], 1)], state_matrix)
#             result += apply_operator_list_on_state([(hop_op_list[1][i][i], 1), 
#                                                     (Iden_u-hop_op_list[0][i][i], 0)], state_matrix)
#             result2 =  apply_operator_list_on_state([(hop_op_list[0][j][j], 0), 
#                                                     (Iden_d-hop_op_list[1][j][j], 1)], result)
#             result2 += apply_operator_list_on_state([(hop_op_list[1][j][j], 1), 
#                                                     (Iden_u-hop_op_list[0][j][j], 0)], result)
#             final_result += result2
#         final_result.shape = dimH
#         return final_result / len(bonds_list)
#     return apply_pp

def create_pp_func(hop_op_list, states_lists, bonds_list):
    dimH_up = np.shape(hop_op_list[0][0][0])[0]
    dimH_down = np.shape(hop_op_list[1][0][0])[0]
    dimH = dimH_up*dimH_down
    pp_op = np.zeros((dimH_up, dimH_down))
    temp  = np.zeros((dimH_up, dimH_down))
    Iden_u = np.ones(dimH_up) 
    Iden_d = np.ones(dimH_down) 

    # pipj = (n_upi*(1-n_dni) + n_dni*(1-n_upi))*(n_upj*(1-n_dnj) + n_dnj*(1-n_upj)) 
    #      = (n_upi + n_dni - 2n_upi*n_dni)*(n_upj + n_dnj - 2*n_upj*n_dnj)
    #      = n_upi*n_upj + n_upi*n_dnj -2*n_upi*n_upj*n_dnj \
    #           + n_dni*n_upj + n_dni*n_dnj - 2*n_dni*n_upj*n_dnj \
    #           - 2*n_upi*n_dni*n_upj -2*n_upi*n_dni*n_dnj + 4 n_upi*n_dni*n_upj*n_dnj
    #       = n_upi*n_upj + n_dni*n_dnj + n_upi*n_dnj + n_dni*n_upj \
    #         -2*n_upi*n_upj*n_dnj - 2*n_dni*n_upj*n_dnj - 2*n_upi*n_dni*n_upj -2*n_upi*n_dni*n_dnj \
    #           + 4 n_upi*n_dni*n_upj*n_dnj

    for i,j in bonds_list:
        np.outer(hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal(), Iden_d, out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        pp_op += temp # n_upi*n_upj

        np.outer(Iden_u, hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        pp_op += temp  #n_dni*n_dnj

        np.outer(hop_op_list[0][i][i].diagonal(), hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        pp_op += temp #n_upi*n_dnj

        np.outer(hop_op_list[0][j][j].diagonal(), hop_op_list[1][i][i].diagonal(), out = temp)
        np.multiply(temp, 1/ len(bonds_list), out=temp)
        pp_op += temp #n_dni*n_upj

        np.outer(hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal(), hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, -2/ len(bonds_list), out=temp)
        pp_op += temp # -2*n_upi*n_upj*n_dnj

        np.outer(hop_op_list[0][j][j].diagonal(),hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, -2/ len(bonds_list), out=temp)
        pp_op += temp # - 2*n_dni*n_upj*n_dnj

        np.outer(hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal(), hop_op_list[1][i][i].diagonal(), out = temp)
        np.multiply(temp, -2/ len(bonds_list), out=temp)
        pp_op += temp # - 2*n_upi*n_dni*n_upj

        np.outer(hop_op_list[0][i][i].diagonal(), hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, -2/ len(bonds_list), out=temp)
        pp_op += temp # -2*n_upi*n_dni*n_dnj 

        np.outer(hop_op_list[0][i][i].diagonal()*hop_op_list[0][j][j].diagonal(), hop_op_list[1][i][i].diagonal()*hop_op_list[1][j][j].diagonal(), out = temp)
        np.multiply(temp, 4/ len(bonds_list), out=temp)
        pp_op += temp # 4*n_upi*n_dni*n_dnj 

    del temp
    def apply_pp(state_matrix):
        # state_matrix = state.reshape(dimH_up, dimH_down)
        result = pp_op * state_matrix
        # result.shape=dimH
        return result
        
    return apply_pp