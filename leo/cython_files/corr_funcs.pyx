import numpy as np
cimport numpy as np
np.import_array()

cdef long get_bit_j_fast(long n, int j):
    cdef long temp = n>>j
    cdef long mask = 1
    cdef long result = temp & mask
    return result

def compute_SzSz_Cloop(psi, int i, int j, states_list):
    cdef double result = 0, resi = 0, resj = 0
    states_up = states_list[0]
    states_down = states_list[1]
    psi_mat = psi.reshape(len(states_up), len(states_down))
    cdef int dimH_up = len(states_up)
    cdef int dimH_down = len(states_down)
    cdef long u = 0, d = 0
    cdef long su, sd, 
    cdef int nui, nuj, ndi, ndj
    cdef double temp = 0.0
    while u < dimH_up:
        su = states_up[u]
        nui = get_bit_j_fast(su, i)
        nuj = get_bit_j_fast(su, j)
        d = 0
        while d < dimH_down:
            sd = states_down[d]
            ndi = get_bit_j_fast(sd, i)
            ndj = get_bit_j_fast(sd, j)
            temp = abs(psi_mat[u,d])**2
            
            result += (nui - ndi)*(nuj-ndj)* temp
            resi   += (nui - ndi)*temp
            resj   += (nuj-ndj)  *temp
            d += 1
        u += 1
    return result - resi * resj


def compute_SzSz_Heisenberg_Cloop(psi, int i, int j, states):
    cdef long D = len(states)
    cdef double result = 0, resi = 0, resj = 0
    cdef long idx = 0
    cdef long a
    cdef int si,sj
    cdef double temp = 0
    while idx < D:
        a = states[idx]
        si = 2*get_bit_j_fast(a, i)-1
        sj = 2*get_bit_j_fast(a, j)-1
        temp = abs(psi[idx])**2
        result += (si)*(sj)*temp
        resi   += (si)*temp
        resj   += (sj)*temp
        idx += 1
    return result - resi * resj


# def compute_SdotS_Cloop(psi, int i, int j, states_list):
#     cdef double result = 0, resi = 0, resj = 0
#     states_up = states_list[0]
#     states_down = states_list[1]
#     psi_mat = psi.reshape(len(states_up), len(states_down))
#     cdef int dimH_up = len(states_up)
#     cdef int dimH_down = len(states_down)
#     cdef long u = 0, d = 0
#     cdef long su, sd, 
#     cdef int nui, nuj, ndi, ndj
#     cdef double temp = 0.0
#     while u < dimH_up:
#         su = states_up[u]
#         nui = get_bit_j_fast(su, i)
#         nuj = get_bit_j_fast(su, j)
#         d = 0
#         while d < dimH_down:
#             sd = states_down[d]
#             ndi = get_bit_j_fast(sd, i)
#             ndj = get_bit_j_fast(sd, j)
#             temp = abs(psi_mat[u,d])**2
            
#             result += (nui - ndi)*(nuj-ndj)* temp
#             resi   += (nui - ndi)*temp
#             resj   += (nuj-ndj)  *temp
#             d += 1
#         u += 1
#     return result - resi * resj


def compute_density_Cloop(psi, int i, states_list):
    cdef double result = 0
    states_up = states_list[0]
    states_down = states_list[1]
    psi_mat = psi.reshape(len(states_up), len(states_down))
    cdef int dimH_up = len(states_up)
    cdef int dimH_down = len(states_down)
    cdef long u = 0, d = 0
    cdef long su, sd, 
    cdef int nui, ndi
    cdef double temp = 0.0
    while u < dimH_up:
        su = states_up[u]
        nui = get_bit_j_fast(su, i)
        d = 0
        while d < dimH_down:
            sd = states_down[d]
            ndi = get_bit_j_fast(sd, i)
            temp = abs(psi_mat[u,d])**2
            result += (nui + ndi)* temp
            d += 1
        u += 1
    return result

def compute_density_corr_Cloop(psi, int i, int j, states_list):
    cdef double result = 0, resi = 0, resj = 0
    states_up = states_list[0]
    states_down = states_list[1]
    psi_mat = psi.reshape(len(states_up), len(states_down))
    cdef int dimH_up = len(states_up)
    cdef int dimH_down = len(states_down)
    cdef long u = 0, d = 0
    cdef long su, sd, 
    cdef int nui, nuj, ndi, ndj
    cdef double temp = 0.0
    while u < dimH_up:
        su = states_up[u]
        nui = get_bit_j_fast(su, i)
        nuj = get_bit_j_fast(su, j)
        d = 0
        while d < dimH_down:
            sd = states_down[d]
            ndi = get_bit_j_fast(sd, i)
            ndj = get_bit_j_fast(sd, j)
            temp = abs(psi_mat[u,d])**2
            
            result += (nui + ndi)*(nuj + ndj)* temp
            resi   += (nui + ndi)* temp
            resj   += (nuj + ndj)* temp
            d += 1
        u += 1
    return result - resi * resj


def compute_HoleSzSz_Cloop(psi, int k, int i, int j, states_list):
    cdef double result = 0, resi = 0, resj = 0
    cdef double resk=0, resij=0,resik=0,resjk=0
    states_up = states_list[0]
    states_down = states_list[1]
    psi_mat = psi.reshape(len(states_up), len(states_down))
    cdef int dimH_up = len(states_up)
    cdef int dimH_down = len(states_down)
    cdef long u = 0, d = 0
    cdef long su, sd
    cdef int nui, nuj, ndi, ndj, nuk, ndk, hk
    cdef double temp = 0.0
    while u < dimH_up:
        su = states_up[u]
        nui = get_bit_j_fast(su, i)
        nuj = get_bit_j_fast(su, j)
        nuk = get_bit_j_fast(su, k)
        d = 0
        while d < dimH_down:
            sd = states_down[d]
            ndi = get_bit_j_fast(sd, i)
            ndj = get_bit_j_fast(sd, j)
            ndk = get_bit_j_fast(sd, k)
            
            temp = abs(psi_mat[u,d])**2
            
            # hk = 1-nuk*ndk
            # dk = nuk & ndk  # bitwise and
            hk = 1 - (nuk | ndk)  # 1- (bitwise or)

            result += hk*(nui-ndi)*(nuj-ndj)* temp
            resi   += (nui-ndi)*temp
            resj   += (nuj-ndj)*temp
            resk   += hk*temp
            resij  += (nui-ndi)*(nuj-ndj)*temp
            resik  += (nui-ndi)*hk*temp
            resjk  += (nuj-ndj)*hk*temp
            d += 1
        u += 1
    return (result - resk*resij - resi*resjk - resj*resik + 2*resi*resj*resk)

def compute_DoublonSzSz_Cloop(psi, int k, int i, int j, states_list):
    cdef double result = 0, resi = 0, resj = 0
    cdef double resk=0, resij=0,resik=0,resjk=0
    states_up = states_list[0]
    states_down = states_list[1]
    psi_mat = psi.reshape(len(states_up), len(states_down))
    cdef int dimH_up = len(states_up)
    cdef int dimH_down = len(states_down)
    cdef long u = 0, d = 0
    cdef long su, sd
    cdef int nui, nuj, ndi, ndj, nuk, ndk, dk
    cdef double temp = 0.0
    while u < dimH_up:
        su = states_up[u]
        nui = get_bit_j_fast(su, i)
        nuj = get_bit_j_fast(su, j)
        nuk = get_bit_j_fast(su, k)
        d = 0
        while d < dimH_down:
            sd = states_down[d]
            ndi = get_bit_j_fast(sd, i)
            ndj = get_bit_j_fast(sd, j)
            ndk = get_bit_j_fast(sd, k)
            
            temp = abs(psi_mat[u,d])**2
            
            # dk = nuk*ndk
            dk = nuk & ndk  # bitwise and
            # hk = 1 - (nuk | ndk)  # 1- (bitwise or)


            result += dk*(nui-ndi)*(nuj-ndj)* temp
            resi   += (nui-ndi)*temp
            resj   += (nuj-ndj)*temp
            resk   += dk*temp
            resij  += (nui-ndi)*(nuj-ndj)*temp
            resik  += (nui-ndi)*dk*temp
            resjk  += (nuj-ndj)*dk*temp
            d += 1
        u += 1
    return (result - resk*resij - resi*resjk - resj*resik + 2*resi*resj*resk)

ctypedef np.float64_t float64_t
ctypedef np.uint32_t uint32_t
def compute_all_correlations_Cloop(psi, int k, int N_sites, states_list):
    cdef np.ndarray[uint32_t] states_up = states_list[0]
    cdef np.ndarray[uint32_t] states_down = states_list[1]
    cdef int dimH_up = len(states_up)
    cdef int dimH_down = len(states_down)
    cdef np.ndarray[float64_t, ndim=2] psi_mat = psi.reshape(dimH_up, dimH_down)
    cdef long u = 0, d = 0
    cdef long su, sd
    cdef int nui, nuj, nuk, ndi, ndj, ndk
    cdef int szi, szj, szk, ni, nj, hk, dk
    cdef double temp = 0.0
    cdef i=0,j=0
    
    cdef double Hk=0, Dk=0
    cdef np.ndarray[float64_t] Sz_list  = np.zeros(N_sites)      # i \in all sites
    cdef np.ndarray[float64_t] N_list   = np.zeros(N_sites)      # i \in all sites
    cdef np.ndarray[float64_t] HSz_list = np.zeros(N_sites)      # k fixed, i \in all sites
    cdef np.ndarray[float64_t] DSz_list = np.zeros(N_sites)      # k fixed, i \in all sites
    cdef np.ndarray[float64_t] NN_list  = np.zeros(N_sites)      # k fixed, i \in all sites
    
    cdef np.ndarray[float64_t, ndim=2] SzSz_list  = np.zeros((N_sites, N_sites))
    cdef np.ndarray[float64_t, ndim=2] HSzSz_list = np.zeros((N_sites, N_sites))
    cdef np.ndarray[float64_t, ndim=2] DSzSz_list = np.zeros((N_sites, N_sites))
    
    while u < dimH_up:
        su = states_up[u]
        nuk = get_bit_j_fast(su, k)
        d = 0
        while d < dimH_down:
            sd = states_down[d]
            temp = np.abs(psi_mat[u,d])**2
            
            ndk = get_bit_j_fast(sd, k)
            dk  = nuk & ndk      # bitwise and
            hk  = 1-(nuk | ndk)   # 1- (bitwise or)
            szk = nuk - ndk
            nk  = nuk + ndk
            
            Dk += dk*temp
            Hk += hk*temp
                
            i=0
            while i<N_sites:
                nui = get_bit_j_fast(su, i)
                ndi = get_bit_j_fast(sd, i)
                szi = nui - ndi
                ni  = nui + ndi
                
                Sz_list[i]   += szi*temp
                N_list[i]    += ni*temp
                NN_list[i]   += nk*ni*temp
                DSz_list[i]  += dk*szi*temp
                HSz_list[i]  += hk*szi*temp
                
                j=0
                while j<N_sites:
                    nuj = get_bit_j_fast(su, j)
                    ndj = get_bit_j_fast(sd, j)
                    szj = nuj - ndj
                    
                    SzSz_list[i,j]  += szi*szj*temp
                    DSzSz_list[i,j] += dk*szi*szj*temp
                    HSzSz_list[i,j] += hk*szi*szj*temp
                    j += 1
                i += 1
            d += 1
        u += 1
    
    #HSS,DSS
    HSS = HSzSz_list - Hk*SzSz_list - np.outer(Sz_list, HSz_list) - np.outer(HSz_list,Sz_list) + 2*Hk*np.outer(Sz_list, Sz_list)
    DSS = DSzSz_list - Dk*SzSz_list - np.outer(Sz_list, DSz_list) - np.outer(DSz_list,Sz_list) + 2*Dk*np.outer(Sz_list, Sz_list)
    
    #SS,NN
    SS = SzSz_list[k,:] - Sz_list[k]*Sz_list
    NN = NN_list - N_list[k]*N_list
    
    return NN, SS, HSS, DSS