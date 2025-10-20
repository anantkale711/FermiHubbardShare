import numpy as np
from uncertainties import unumpy
import time
import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt

def compute_emin_emax_dicts(result_evals, seeds):
    emin_dict = {}
    emax_dict = {}
    N_list = list(seeds.keys())
    for (nu, nd) in N_list:
        emin, emax = compute_emin_emax_single(result_evals[(nu,nd)], seeds[(nu,nd)])
        emin_dict[(nu, nd)] = emin
        emax_dict[(nu, nd)] = emax
    return emin_dict, emax_dict    

def compute_emin_emax_single(result_evals, seeds):
    if len(seeds)==1:
        emin = np.min(result_evals[0])
        emax = np.max(result_evals[0])
    else:
        emin = np.min([np.min(evals_ri) for evals_ri in result_evals])
        emax = np.max([np.max(evals_ri[evals_ri != 1e6]) for evals_ri in result_evals])
    return emin, emax

def compute_thermal_average_sector_resolved(beta_list, dimH_dict, result_evals, result_evecs, result, seeds):
    '''
    params: 
    beta_list: list of inverse temperatures 
    dimH_dict: dict containing Hilbert space dimensions of the various sectors 
    result_evals, result_evecs, result, seeds
    returns:
    Z_mega_list: dict of partition functions (vs beta) for each sector
    thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    emin_dict: dict of ground state energies for each sector
    '''
    emin_dict, _ = compute_emin_emax_dicts(result_evals, seeds)
    N_list = list(seeds.keys())
    N_quantities = len(result[N_list[0]][0])
    thermal_avg_q = [{} for qi in range(N_quantities)]
    Z_mega_list  = {} # partition function
    

    for (nu, nd) in N_list:
        Z_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        for qi in range(N_quantities):
            thermal_avg_q[qi][(nu,nd)] = unumpy.uarray(np.zeros(len(beta_list)),np.zeros(len(beta_list)))

    tic_start = time.time()
    for (nu, nd) in tqdm.tqdm(N_list):
        emin = emin_dict[(nu,nd)]
        N_seeds = len(seeds[(nu,nd)])
        if N_seeds>1:
            Z_r = []
            q_r = [[] for qi in range(N_quantities)]
            #  print(f_i, sztot_i, u_i)
            for r_i in range(N_seeds):
                evals = result_evals[(nu, nd)][r_i]
                evecs = result_evecs[(nu, nd)][r_i]
                evals2 = np.copy(result_evals[(nu, nd)][r_i])
                idx2 = (evals2 != 1e6)
                # evals2[evals2 == 1e6] = U/2*(nu+nd)
#                 Z = np.array([np.sum(np.abs(evecs[0,:])**2 * np.exp(-beta*(evals-emin))) for beta in beta_list])
                Z = np.sum(np.abs(evecs[0,idx2, np.newaxis])**2 * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
                Z_r.append(Z)
                
                for qi in range(N_quantities):
                    overlaps_q = result[(nu, nd)][r_i][qi]
#                     overlaps_q = results[qi][f_i][sztot_i][u_i][r_i]
#                     q_temp = np.array([np.sum(evecs[0,:] * np.exp(-beta*(evals-emin))* (np.conjugate(evecs.T) @ overlaps_q)) for beta in beta_list])
                    q_temp = np.sum(evecs[0,idx2, np.newaxis] * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin))* (np.conjugate(evecs.T) @ overlaps_q)[idx2,np.newaxis], axis=0) 
                    q_r[qi].append(q_temp)
                    
            ZR = unumpy.uarray(np.mean(Z_r, axis=0),np.std(Z_r, axis=0)/np.sqrt(N_seeds))
            Z_mega_list[(nu, nd)] = dimH_dict[(nu, nd)] * ZR

            for qi in range(N_quantities):
                qR = unumpy.uarray(np.mean(q_r[qi], axis=0), np.std(q_r[qi], axis=0)/np.sqrt(N_seeds))
                thermal_avg_q[qi][(nu, nd)]  = dimH_dict[(nu, nd)] * qR / Z_mega_list[(nu, nd)]

        else:
            Z_r = np.zeros(len(beta_list))
            q_r = [np.zeros(len(beta_list)) for qi in range(N_quantities)]
            r_i = 0
            evals = result_evals[(nu,nd)][0]
            # m, beta
            Z_r = np.sum(np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
            Z_mega_list[(nu,nd)] = unumpy.uarray(Z_r, np.zeros_like(Z_r))
            
            for qi in range(N_quantities):
                overlaps_q = result[(nu,nd)][0][qi]
                q_temp = np.sum(np.exp(-beta_list[np.newaxis,:]*(evals[:, np.newaxis]-emin))* (overlaps_q[:, np.newaxis]), axis=0) 
                q_r[qi] = q_temp / Z_r
                thermal_avg_q[qi][(nu,nd)] = unumpy.uarray(q_r[qi], np.zeros_like(q_r[qi]))
    

    toc_stop = time.time()
    print(f"Total time taken: {toc_stop-tic_start:.2f} seconds")
    return Z_mega_list, thermal_avg_q, emin_dict


def compute_grand_average(mu_list, beta_list, U, scale2, Z_mega_list, thermal_avg_q, emin_dict):
    '''
    params:
    mu_list: list of chemical potentials
    beta_list: list of inverse temperatures
    U: interaction strength
    scale2: coefficients used for managing exponential blow up of Z
    Z_mega_list: dict of partition functions (vs beta) for each sector
    thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    emin_dict: dict of ground state energies for each sector

    returns:
    Zmu_list: 2D unumpy.uarray containing grand potential Z(mu, beta)  
    grand_avg_q: list of 2D uarrays containing quantities A(mu, beta) 
    '''
    
    N_quantities = len(thermal_avg_q)
    N_list = list(emin_dict.keys())
    N_sites = np.max(N_list)
    
    Zmu_vals = np.zeros((len(mu_list), len(beta_list))) 
    Zmu_variance = np.zeros((len(mu_list), len(beta_list))) 
    Zmu_errs = np.zeros((len(mu_list), len(beta_list))) 

    grand_avg_q_vals = [np.zeros((len(mu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_variance = [np.zeros((len(mu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_errs = [np.zeros((len(mu_list), len(beta_list))) for qi in range(N_quantities)]

    scale = 1.0
    
    tic_start = time.time()  
    for m_i, mu in enumerate(tqdm.tqdm(mu_list)):
        for (nu,nd) in N_list:
            log_factor =  beta_list*((mu+U/2)*(nu+nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2))
            Zmu_vals[m_i] += unumpy.nominal_values(Z_mega_list[(nu,nd)][:])* np.exp(log_factor)
            Zmu_variance[m_i] += unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2 *np.exp(2*log_factor)
            
        Zmu_errs[m_i] = np.sqrt(Zmu_variance[m_i])
        logZmu = np.log(Zmu_vals[m_i])
        
        for (nu,nd) in N_list:
            log_factor =  beta_list*((mu+U/2)*(nu+nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2))
            ZbetaN = unumpy.nominal_values(Z_mega_list[(nu,nd)][:])
            logZbetaN = np.log(ZbetaN)
                   
            for q_i in range(N_quantities):
                grand_avg_q_vals[q_i][m_i,  :] += np.exp(log_factor - logZmu + logZbetaN)*\
                                                unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])
                grand_avg_q_variance[q_i][m_i,  :] += np.exp(2*log_factor - 2*logZmu)*\
                                                unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])**2 *\
                                                unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
                
                grand_avg_q_variance[q_i][m_i,  :] += np.exp(2*log_factor - 2*logZmu + 2*logZbetaN)*\
                                                unumpy.std_devs(thermal_avg_q[q_i][(nu,nd)][:])**2
            
                    
        for q_i in range(N_quantities):
            grand_avg_q_variance[q_i][m_i,  :] += Zmu_errs[m_i]**2 *(grand_avg_q_vals[q_i][m_i])**2 * np.exp(- 2*logZmu)
            grand_avg_q_errs[q_i][m_i,  :] = np.sqrt(grand_avg_q_variance[q_i][m_i, :])
        
    Zmu_list = unumpy.uarray(Zmu_vals, Zmu_errs)
    grand_avg_q = [unumpy.uarray(grand_avg_q_vals[qi], grand_avg_q_errs[qi]) for qi in range(N_quantities)]
    print(f"Total time taken: {time.time()-tic_start:.2f} seconds")
    return Zmu_list, grand_avg_q


def compute_grand_average_with_Bfield(dmu_list, beta_list, mu, U, scale2, scale3, Z_mega_list, thermal_avg_q, emin_dict):
    '''
    params:
    dmu_list: list of chemical potential difference applied to up and down spins
    beta_list: list of inverse temperatures
    mu: chemical potential value to be used 
    U: interaction strength
    scale2, scale3: coefficients used for managing exponential blow up of Z
    Z_mega_list: dict of partition functions (vs beta) for each sector
    thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    emin_dict: dict of ground state energies for each sector

    returns:
    Zdmu_list: 2D unumpy.uarray containing grand potential Z(dmu, beta)  
    grand_avg_q2: list of 2D uarrays containing quantities A(dmu, beta) 
    '''

    N_quantities = len(thermal_avg_q)
    N_list = list(emin_dict.keys())
    N_sites = np.max(N_list)
    
    Zdmu_vals = np.zeros((len(dmu_list), len(beta_list))) 
    Zdmu_variance = np.zeros((len(dmu_list), len(beta_list))) 
    Zdmu_errs = np.zeros((len(dmu_list), len(beta_list))) 
    grand_avg_q_vals2 = [np.zeros((len(dmu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_variance2 = [np.zeros((len(dmu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_errs2 = [np.zeros((len(dmu_list), len(beta_list))) for qi in range(N_quantities)]

    scale = 1.0
    tic_start = time.time()  
    for dm_i, dmu in enumerate(tqdm.tqdm(dmu_list)):
        tic = time.time()    
        for (nu,nd) in N_list:
            log_factor =  beta_list*((mu+dmu/2+U/2)*(nu) + (mu-dmu/2+U/2)*(nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2)- np.exp(abs(dmu) *scale3))
            Zdmu_vals[dm_i] += unumpy.nominal_values(Z_mega_list[(nu,nd)][:])* np.exp(log_factor)
            Zdmu_variance[dm_i] += unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2 *np.exp(2*log_factor)
        Zdmu_errs[dm_i] = np.sqrt(Zdmu_variance[dm_i])
        logZdmu = np.log(Zdmu_vals[dm_i])
        for (nu,nd) in N_list:
            log_factor =  beta_list*((mu+dmu/2+U/2)*(nu) + (mu-dmu/2+U/2)*(nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2) - np.exp(abs(dmu) * scale3))
            ZbetaN = unumpy.nominal_values(Z_mega_list[(nu,nd)][:])
            logZbetaN = np.log(ZbetaN)
            for q_i in range(N_quantities):
                grand_avg_q_vals2[q_i][dm_i,  :] += np.exp(log_factor - logZdmu + logZbetaN)*\
                                                unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])
                grand_avg_q_variance2[q_i][dm_i,  :] += np.exp(2*log_factor - 2*logZdmu)*\
                                                unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])**2 *\
                                                unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
                grand_avg_q_variance2[q_i][dm_i,  :] += np.exp(2*log_factor - 2*logZdmu + 2*logZbetaN)*\
                                                unumpy.std_devs(thermal_avg_q[q_i][(nu,nd)][:])**2
                
        for q_i in range(N_quantities):
            grand_avg_q_variance2[q_i][dm_i,  :] += Zdmu_errs[dm_i]**2 *(grand_avg_q_vals2[q_i][dm_i])**2 * np.exp(- 2*logZdmu)
            grand_avg_q_errs2[q_i][dm_i,  :] = np.sqrt(grand_avg_q_variance2[q_i][dm_i, :])
        
    Zdmu_list = unumpy.uarray(Zdmu_vals, Zdmu_errs)
    grand_avg_q2 = [unumpy.uarray(grand_avg_q_vals2[qi], grand_avg_q_errs2[qi]) for qi in  range(N_quantities)]
    print(f"Total time taken: {time.time()-tic_start:.2f} seconds")
    return Zdmu_list, grand_avg_q2

def compute_thermal_average_sector_single_v2(beta_list, dimH, result_evals, result_evecs, result, seeds, U, N_tot):
    '''
    params: 
    beta_list: list of inverse temperatures 
    dimH: Hilbert space dimension of the given sector 
    result_evals, result_evecs, result, seeds
    U, N_tot: interaction U/t and total number of particles (needed for computing correct energy)
    returns:
    thermo_dict: dict of Z,E,F,S (vs beta) for given sector (partition function, internal energy, free energy, entropy)
    thermal_avg_q: list of thermal avgs of the various quantities in given sector
    emin: ground state energy for given sector
    '''
    emin, _ = compute_emin_emax_single(result_evals, seeds)
    N_quantities = len(result[0])
    Z_mega_list  = [] # partition function
    F_mega_list = []  # free energy
    E_mega_list = []  # average energy
    S_mega_list = []  # entropy

    Z_mega_list   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
    F_mega_list   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
    E_mega_list   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
    S_mega_list   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
    thermal_avg_q = []
    for qi in range(N_quantities):
        thermal_avg_q.append(unumpy.uarray(np.zeros(len(beta_list)),np.zeros(len(beta_list))))

    
    N_seeds = len(seeds)
    if N_seeds>1:
        Z_r = []
        E_r = []
        q_r = [[] for qi in range(N_quantities)]
        #  print(f_i, sztot_i, u_i)
        for r_i in range(N_seeds):
            evals = result_evals[r_i]
            evecs = result_evecs[r_i]
            evals2 = np.copy(result_evals[r_i])
            idx2 = (evals2 != 1e6)
            Z = np.sum(np.abs(evecs[0,idx2, np.newaxis])**2 * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
            Z_r.append(Z)
            E = np.sum((evals2[idx2,np.newaxis]-U/2*N_tot) * np.abs(evecs[0,idx2, np.newaxis])**2 * \
                                                            np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
            E_r.append(E)

            for qi in range(N_quantities):
                overlaps_q = result[r_i][qi]
                q_temp = np.sum(evecs[0,idx2, np.newaxis] \
                                * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin))\
                                    * (np.conjugate(evecs.T) @ overlaps_q)[idx2,np.newaxis], 
                                    axis=0) 
                q_r[qi].append(q_temp)
                
        ZR = unumpy.uarray(np.mean(Z_r, axis=0),np.std(Z_r, axis=0)/np.sqrt(N_seeds))
        Z_mega_list = dimH * ZR

        ER = unumpy.uarray(np.mean(E_r, axis=0),np.std(E_r, axis=0)/np.sqrt(N_seeds))
        E_mega_list = dimH * ER / Z_mega_list

        FR_v = -1/beta_list*(np.log(unumpy.nominal_values(Z_mega_list)) - beta_list*(emin-U/2*N_tot))
        FR_e = 1/beta_list*unumpy.std_devs(Z_mega_list)/unumpy.nominal_values(Z_mega_list)
        F_mega_list = unumpy.uarray(FR_v, FR_e)
        S_mega_list = beta_list*(E_mega_list-F_mega_list)

        for qi in range(N_quantities):
            qR = unumpy.uarray(np.mean(q_r[qi], axis=0), np.std(q_r[qi], axis=0)/np.sqrt(N_seeds))
            thermal_avg_q[qi]  = dimH * qR / Z_mega_list

    else:
        Z_r = np.zeros(len(beta_list))
        E_r = np.zeros(len(beta_list))
        q_r = [np.zeros(len(beta_list)) for qi in range(N_quantities)]
        r_i = 0
        evals = result_evals[0]
        # m, beta
        Z_r = np.sum(np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
        Z_mega_list = unumpy.uarray(Z_r, np.zeros_like(Z_r))
        E_r = np.sum((evals[:, np.newaxis]-U/2*N_tot) * np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
        E_mega_list = unumpy.uarray(E_r/Z_r, np.zeros_like(E_r))
        F_mega_list = unumpy.uarray(-1/beta_list*(np.log(Z_r)- beta_list*(emin-U/2*(N_tot))), np.zeros_like(Z_r))
        S_mega_list = beta_list*(E_mega_list-F_mega_list)

        for qi in range(N_quantities):
            overlaps_q = result[0][qi]
            q_temp = np.sum(np.exp(-beta_list[np.newaxis,:]*(evals[:, np.newaxis]-emin))* (overlaps_q[:, np.newaxis]), axis=0) 
            q_r[qi] = q_temp / Z_r
            thermal_avg_q[qi] = unumpy.uarray(q_r[qi], np.zeros_like(q_r[qi]))
    thermal_avg_q.append(E_mega_list)  # add energy to the list of quantities
    thermo_dict = {'Z':Z_mega_list, 'E':E_mega_list, 'F':F_mega_list, 'S':S_mega_list}
    return thermo_dict, thermal_avg_q, emin


def compute_thermal_average_LTLM_sector_single(beta_list, dimH, result_evals, result_evecs, result, seeds, U, N_tot):
    '''
    params: 
    beta_list: list of inverse temperatures 
    dimH: Hilbert space dimension of the given sector 
    result_evals, result_evecs, result, seeds
    U, N_tot: interaction U/t and total number of particles (needed for computing correct energy)
    returns:
    thermo_dict: dict of Z,E,F,S (vs beta) for given sector (partition function, internal energy, free energy, entropy)
    thermal_avg_q: list of thermal avgs of the various quantities in given sector
    emin: ground state energy for given sector
    '''
    emin, _ = compute_emin_emax_single(result_evals, seeds)
    N_quantities = len(result[0])
    Z_mega_list  = [] # partition function
    F_mega_list = []  # free energy
    E_mega_list = []  # average energy
    S_mega_list = []  # entropy

    Z_mega_list   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
    F_mega_list   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
    E_mega_list   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
    S_mega_list   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
    thermal_avg_q = []
    for qi in range(N_quantities):
        thermal_avg_q.append(unumpy.uarray(np.zeros(len(beta_list)),np.zeros(len(beta_list))))

    
    N_seeds = len(seeds)
    if N_seeds>1:
        N_states = len(result_evals[0])
        Z_r = np.zeros((N_seeds, len(beta_list)))
        E_r = np.zeros((N_seeds, len(beta_list)))
        q_r = [np.zeros((N_seeds, len(beta_list))) for qi in range(N_quantities)]
        #  print(f_i, sztot_i, u_i)
        for r_i in range(N_seeds):
            evals = result_evals[r_i]
            evecs = result_evecs[r_i]
            evals2 = np.copy(result_evals[r_i])
            idx2 = (evals2 != 1e6)
            Z = np.sum(np.abs(evecs[0,idx2, np.newaxis])**2 * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
            Z_r[r_i] = Z
            E = np.sum((evals2[idx2,np.newaxis]-U/2*N_tot) * np.abs(evecs[0,idx2, np.newaxis])**2 * \
                                                            np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
            E_r[r_i] = E

            exp_factor_temp =  np.exp(-beta_list[np.newaxis, np.newaxis, :]*(evals[idx2, np.newaxis, np.newaxis]-emin + evals[np.newaxis, idx2, np.newaxis]-emin)/2)
            # print(exp_factor_temp.shape)

            for qi in range(N_quantities):
                overlaps_q = result[r_i][qi]
                # overlaps_q_eigbasis = np.einsum('mn,nl,mj->lj', overlaps_q, evecs[:,idx2], np.conjugate(evecs[:,idx2].T))
                overlaps_q_eigbasis = np.einsum('mn,nl,mj->lj', overlaps_q, evecs[:,idx2], np.conjugate(evecs[:,idx2]))

                
                q_temp = np.einsum('jlb,j,l,lj->b', 
                                        exp_factor_temp,
                                        evecs[0,idx2],
                                        np.conjugate(evecs[0,idx2]),
                                        overlaps_q_eigbasis ) 
                q_r[qi][r_i] = q_temp
                
        ZR = unumpy.uarray(np.mean(Z_r, axis=0),np.std(Z_r, axis=0)/np.sqrt(N_seeds))
        Z_mega_list = dimH * ZR

        ER = unumpy.uarray(np.mean(E_r, axis=0),np.std(E_r, axis=0)/np.sqrt(N_seeds))
        E_mega_list = dimH * ER / Z_mega_list

        FR_v = -1/beta_list*(np.log(unumpy.nominal_values(Z_mega_list)) - beta_list*(emin-U/2*N_tot))
        FR_e = 1/beta_list*unumpy.std_devs(Z_mega_list)/unumpy.nominal_values(Z_mega_list)
        F_mega_list = unumpy.uarray(FR_v, FR_e)
        S_mega_list = beta_list*(E_mega_list-F_mega_list)

        # for qi in range(N_quantities):
        #     qR = unumpy.uarray(np.mean(q_r[qi], axis=0), np.std(q_r[qi], axis=0)/np.sqrt(N_seeds))
        #     thermal_avg_q[qi]  = dimH * qR / Z_mega_list

        for qi in range(N_quantities):
            vals = np.zeros((N_seeds, len(beta_list)))
            for ri in range(N_seeds):
                idxs = (np.arange(N_seeds) != ri)
                q_temp = q_r[qi][idxs]
                Z_temp = Z_r[idxs]
                vals[ri]= np.sum(q_temp, axis=0)/np.sum(Z_temp, axis=0)
            
            jack_mean = np.sum(vals, axis=0)/N_seeds
            jack_std = np.sqrt((N_seeds-1)/N_seeds*np.sum((vals-jack_mean)**2, axis=0)) 
            qR = unumpy.uarray(jack_mean, jack_std)
            thermal_avg_q[qi]  = qR


    else:
        Z_r = np.zeros(len(beta_list))
        E_r = np.zeros(len(beta_list))
        q_r = [np.zeros(len(beta_list)) for qi in range(N_quantities)]
        r_i = 0
        evals = result_evals[0]
        # m, beta
        Z_r = np.sum(np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
        Z_mega_list = unumpy.uarray(Z_r, np.zeros_like(Z_r))
        E_r = np.sum((evals[:, np.newaxis]-U/2*N_tot) * np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
        E_mega_list = unumpy.uarray(E_r/Z_r, np.zeros_like(E_r))
        F_mega_list = unumpy.uarray(-1/beta_list*(np.log(Z_r)- beta_list*(emin-U/2*(N_tot))), np.zeros_like(Z_r))
        S_mega_list = beta_list*(E_mega_list-F_mega_list)

        for qi in range(N_quantities):
            overlaps_q = result[0][qi]
            q_temp = np.sum(np.exp(-beta_list[np.newaxis,:]*(evals[:, np.newaxis]-emin))* (overlaps_q[:, np.newaxis]), axis=0) 
            q_r[qi] = q_temp / Z_r
            thermal_avg_q[qi] = unumpy.uarray(q_r[qi], np.zeros_like(q_r[qi]))
    thermal_avg_q.append(E_mega_list)  # add energy to the list of quantities
    thermo_dict = {'Z':Z_mega_list, 'E':E_mega_list, 'F':F_mega_list, 'S':S_mega_list}
    return thermo_dict, thermal_avg_q, emin


# def compute_dynamical_response_FTLM_sector_single(omega_list, beta, dimH, result_dicts, seeds, U, N_tot):
#     '''
#     params: 
#     omega_list: list of real frequencies 
#     beta: inverse temperature
#     dimH: Hilbert space dimension of the given sector 
#     result_dict, seeds
#     U, N_tot: interaction U/t and total number of particles (needed for computing correct energy)
#     returns:
#     response_real: 
#     response_imag: 
#     emin: ground state energy for given sector
#     '''
#     N_seeds = len(seeds)
#     print(N_seeds)
#     if N_seeds == 1:
#         emin = np.min(result_dicts[0]['evals'])
#     else:
#         emin = np.min([np.min(result_dicts[i]['evals']) for i in range(N_seeds)])
    
#     def lor(x,eps=0.2):
#         return 1/np.pi*eps/(x**2 + eps**2)

#     def pole(x,eps=0.5):
#         return 1/(x+1j*eps)
    
#     if N_seeds>1:        
#         #  print(f_i, sztot_i, u_i)
#         vals_r = np.zeros((N_seeds, len(omega_list)), dtype=complex)
#         Z_r = np.zeros(N_seeds)

#         for r_i in range(N_seeds):
#             evecs = result_dicts[r_i]['evecs']
#             evecs1 = result_dicts[r_i]['evecs1']
#             evals = result_dicts[r_i]['evals']
#             evals1 = result_dicts[r_i]['evals1']

#             Z = np.sum(np.abs(evecs[0,:])**2 * np.exp(-beta*(evals[:]-emin)), axis=0)
#             Z_r[r_i] = Z
            
#             energy_ij = evals[:,np.newaxis]-evals1[np.newaxis,:]
#             delta_function = pole(omega_list[np.newaxis,np.newaxis,:]+energy_ij[:,:,np.newaxis])
#             exp_factor_i = np.exp(-beta*(evals[:]-emin))
#             # overlaps = result_dicts[r_i]['overlaps'].T*result_dicts[r_i]['state_norm']
#             # overlaps_ij = np.einsum('ik,kl,lj->ij',evecs[:,:], overlaps, evecs1.T[:,:],optimize=True)
#             overlaps_kl = result_dicts[r_i]['overlaps']*result_dicts[r_i]['state_norm']
#             overlaps_ij = np.einsum('ki,kl,lj->ij',evecs[:,:], overlaps_kl, evecs1[:,:],optimize=True)
#             sum_ij = np.einsum('i,ij,j->ij', evecs[0,:], overlaps_ij, evecs1[:,0], optimize=True)
            
#             q_temp = np.einsum('i,ijw,ij->w', exp_factor_i, delta_function, sum_ij, out=vals_r[r_i], optimize=True)


#         jack_vals_r = np.zeros((N_seeds, len(omega_list)))
#         jack_vals_i = np.zeros((N_seeds, len(omega_list)))
#         jack_Z = np.zeros(N_seeds)
#         for ri in range(N_seeds):
#             idxs = (np.arange(N_seeds) != ri)
#             q_temp1 = (vals_r.real)[idxs]
#             q_temp2 = (vals_r.imag)[idxs]
#             Z_temp = Z_r[idxs]
#             jack_Z[ri] = np.mean(Z_temp)
#             jack_vals_r[ri]= np.sum(q_temp1, axis=0)/np.sum(Z_temp, axis=0)
#             jack_vals_i[ri]= np.sum(q_temp2, axis=0)/np.sum(Z_temp, axis=0)

#         jack_mean_r = np.sum(jack_vals_r, axis=0)/N_seeds
#         jack_mean_i = np.sum(jack_vals_i, axis=0)/N_seeds

#         jack_std_r = np.sqrt((N_seeds-1)/N_seeds*np.sum((jack_vals_r-jack_mean_r)**2, axis=0))
#         jack_std_i = np.sqrt((N_seeds-1)/N_seeds*np.sum((jack_vals_i-jack_mean_i)**2, axis=0)) 

#         jack_Z_mean = np.mean(jack_Z)
#         Z_avg = unumpy.uarray(jack_Z_mean, np.sqrt((N_seeds-1)/N_seeds*np.sum((jack_Z-jack_Z_mean)**2)))
#         response_real = unumpy.uarray(jack_mean_r, jack_std_r)
#         response_imag = unumpy.uarray(jack_mean_i, jack_std_i)

#     else:
        
#         r_i = 0
#         evals = result_dicts[0]['evals']
#         Z = np.sum(np.exp(-beta*(evals-emin)))
        
#         energy_ij = evals[:,np.newaxis]-evals[np.newaxis,:]
#         delta_function = lor(omega_list[:,np.newaxis,np.newaxis]+energy_ij[np.newaxis,:,:])
#         exp_factor_i = np.exp(-beta*(evals[:]-emin))
            
#         overlaps = result_dicts[0]['overlaps']**2
#         # result = np.einsum('ij,wij,i->w', overlaps, delta_function, exp_factor)
        
#         q_temp = np.einsum('ij,i->ij', overlaps, exp_factor_i)
#         result = np.einsum('wij,ij->w', delta_function,q_temp)
#         response_real = unumpy.uarray(np.real(result/Z), np.zeros(len(omega_list)))
#         response_imag = unumpy.uarray(np.imag(result/Z), np.zeros(len(omega_list)))
#         Z_avg = unumpy.uarray(Z, np.zeros(len(omega_list)))

#     return response_real, response_imag, Z_avg*dimH, emin


def compute_dynamical_response_FTLM_sector_single(omega_list, t1, beta, dimH, result_dicts, seeds, U, N_tot):
    '''
    params: 
    omega_list: list of real frequencies 
    beta: inverse temperature
    dimH: Hilbert space dimension of the given sector 
    result_dict, seeds
    U, N_tot: interaction U/t and total number of particles (needed for computing correct energy)
    returns:
    response_real: 
    response_imag: 
    emin: ground state energy for given sector
    '''
    N_seeds = len(seeds)
    print(N_seeds)
    if N_seeds == 1:
        emin = np.min(result_dicts[0]['evals'])
    else:
        emin = np.min([np.min(result_dicts[i]['evals']) for i in range(N_seeds)])
    
    dt = t1[1]-t1[0]
    # sigma=5
    # envelope = np.exp(-t1**2/sigma)
    # envelope /= np.sum(envelope*dt)

    if N_seeds>1:        
        #  print(f_i, sztot_i, u_i)
        vals_w_r = np.zeros((N_seeds, len(omega_list)), dtype=complex)
        vals_t_r = np.zeros((N_seeds, len(omega_list)), dtype=complex)
        Z_r = np.zeros(N_seeds)

        for r_i in range(N_seeds):
            U = result_dicts[r_i]['evecs']
            U1 = result_dicts[r_i]['evecs1']
            E = result_dicts[r_i]['evals']
            E1 = result_dicts[r_i]['evals1']

            Z = np.sum(np.abs(U[0,:])**2 * np.exp(-beta*(E[:]-emin)), axis=0)
            Z_r[r_i] = Z
            
            V_k = np.einsum('ti,i,ki->tk', np.exp((1j*t1[:,np.newaxis]-beta)*(E[np.newaxis,:]-emin)),U[0,:], U[:,:], optimize=True)
            
            V1_l = np.einsum('tj,lj,j->tl', np.exp((-1j*t1[:,np.newaxis])*(E1[np.newaxis,:]-emin)),U1[:,:], U1[0,:], optimize=True)

            overlaps_kl = result_dicts[r_i]['overlaps']*result_dicts[r_i]['state_norm']
            res_t = np.einsum('tk,kl,tl->t',V_k, overlaps_kl, V1_l, optimize=True)
            # res_t *= envelope
            
            res_omega = np.einsum('wt,t->w', np.exp(1j*(omega_list[:, np.newaxis]+1j*0.1)*t1[np.newaxis, :]), res_t)*dt
            vals_w_r[r_i] = res_omega
            vals_t_r[r_i] = res_t

        jack_vals_w_r = np.zeros((N_seeds, len(omega_list)))
        jack_vals_w_i = np.zeros((N_seeds, len(omega_list)))
        jack_vals_t_r = np.zeros((N_seeds, len(t1)))
        jack_vals_t_i = np.zeros((N_seeds, len(t1)))
        jack_Z = np.zeros(N_seeds)
        for ri in range(N_seeds):
            idxs = (np.arange(N_seeds) != ri)
            Z_temp = Z_r[idxs]
            jack_Z[ri] = np.mean(Z_temp)
            q_temp_r = (vals_w_r.real)[idxs]
            q_temp_i = (vals_w_r.imag)[idxs]
            jack_vals_w_r[ri]= np.sum(q_temp_r, axis=0)/np.sum(Z_temp, axis=0)
            jack_vals_w_i[ri]= np.sum(q_temp_i, axis=0)/np.sum(Z_temp, axis=0)
            q_temp_r = (vals_t_r.real)[idxs]
            q_temp_i = (vals_t_r.imag)[idxs]
            jack_vals_t_r[ri]= np.sum(q_temp_r, axis=0)/np.sum(Z_temp, axis=0)
            jack_vals_t_i[ri]= np.sum(q_temp_i, axis=0)/np.sum(Z_temp, axis=0)

        jack_mean_w_r = np.sum(jack_vals_w_r, axis=0)/N_seeds
        jack_mean_w_i = np.sum(jack_vals_w_i, axis=0)/N_seeds
        jack_mean_t_r = np.sum(jack_vals_t_r, axis=0)/N_seeds
        jack_mean_t_i = np.sum(jack_vals_t_i, axis=0)/N_seeds

        jack_std_w_r = np.sqrt((N_seeds-1)/N_seeds*np.sum((jack_vals_w_r-jack_mean_w_r)**2, axis=0))
        jack_std_w_i = np.sqrt((N_seeds-1)/N_seeds*np.sum((jack_vals_w_i-jack_mean_w_i)**2, axis=0)) 
        jack_std_t_r = np.sqrt((N_seeds-1)/N_seeds*np.sum((jack_vals_t_r-jack_mean_t_r)**2, axis=0))
        jack_std_t_i = np.sqrt((N_seeds-1)/N_seeds*np.sum((jack_vals_t_i-jack_mean_t_i)**2, axis=0)) 

        jack_Z_mean = np.mean(jack_Z)
        Z_avg = unumpy.uarray(jack_Z_mean, np.sqrt((N_seeds-1)/N_seeds*np.sum((jack_Z-jack_Z_mean)**2)))
        response_w_real = unumpy.uarray(jack_mean_w_r, jack_std_w_r)
        response_w_imag = unumpy.uarray(jack_mean_w_i, jack_std_w_i)
        response_t_real = unumpy.uarray(jack_mean_t_r, jack_std_t_r)
        response_t_imag = unumpy.uarray(jack_mean_t_i, jack_std_t_i)

    else:
        
        r_i = 0
        evals = result_dicts[0]['evals']
        Z = np.sum(np.exp(-beta*(evals-emin)))
        overlaps = result_dicts[0]['overlaps']**2
        
        C_jt = np.einsum('ij,it->jt', overlaps, np.exp((-beta+1j*t1[np.newaxis,:])*(evals[:,np.newaxis]-emin)))
        C_t = np.einsum('jt,jt->t', C_jt, np.exp(-1j*t1[np.newaxis, :]*(evals[:, np.newaxis]-emin)), optimize=True)
        C_omega = np.einsum('wt,t->w', np.exp(1j*(omega_list[:, np.newaxis]+1j*0.1)*t1[np.newaxis, :]), C_t)*dt
            
        response_w_real = unumpy.uarray(np.real(C_omega/Z), np.zeros(len(omega_list)))
        response_w_imag = unumpy.uarray(np.imag(C_omega/Z), np.zeros(len(omega_list)))        
        response_t_real = unumpy.uarray(np.real(C_t/Z), np.zeros(len(t1)))
        response_t_imag = unumpy.uarray(np.imag(C_t/Z), np.zeros(len(t1)))        
        Z_avg = unumpy.uarray(Z, 0)

    return response_w_real, response_w_imag, response_t_real, response_t_imag, Z_avg*dimH, emin

def process_sector_single(raw_data_dir, save_dir, sector_dict, sector_no, beta_list=None, isLTLM=False):
    
    tic0 = time.time()
    N_sites, N_up,N_down,dimH = sector_dict[sector_no]
    N_tot = N_up + N_down

    
    U = 0
    result = []
    seeds  = []
    result_evals = []
    result_evecs = []
    keys_to_array = None
    data_dir = raw_data_dir
    
    all_files = os.listdir(data_dir)
    print("Found %d files in data_dir folder"%(len(all_files)))
    tic = time.time()
    for filename in all_files:
        if filename.endswith(".npz"):
            pars = filename[:-4].split('_')[1::2]
            nup = int(pars[0])
            ndown = int(pars[1])
            U = float(pars[2])
            seed = int(pars[3]) 
            if nup != N_up or ndown != N_down:
                continue
            full_path = os.path.join(data_dir, filename)
            with np.load(full_path) as dfile:
                if keys_to_array is None:
                    keys_to_array = []
                    for k in sorted(dfile.files):
                        if k not in ["evecs", "evals"]:
                            keys_to_array.append(k)
                temp_lst = np.array([dfile[k] for k in keys_to_array])
                result_evals.append(dfile["evals"])
                if seed != 0:
                    result_evecs.append(dfile["evecs"])
                result.append(temp_lst)
                seeds.append(seed)
    toc = time.time()
    print("Loaded %d files of sector number %d in %.2f seconds"%(len(seeds), sector_no, toc-tic))

    print(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(save_dir, "Nup_%d_Ndown_%d_quantities_list.txt"%(N_up, N_down))
    with open(filename, 'w') as f:
        for key in keys_to_array:
            f.write(f"{key}\n")
    
    if beta_list is None:
        if isLTLM:
            beta_list = np.linspace(25, 5, 21)
            beta_list = np.array(list(beta_list) + list(np.linspace(5, 0.01, 11)))
        else:
            beta_list = 1/np.linspace(0.05, 1.05, 21)
            beta_list = np.array(list(beta_list) + list(np.linspace(0.67, 0.01, 11)))
    
    # beta_list = np.linspace(50, 10, 21)
    # beta_list = np.array(list(beta_list) + list(np.linspace(10, 0.1, 11)))

    filename = os.path.join(save_dir, "Nup_%d_Ndown_%d_T_list.txt"%(N_up, N_down))
    with open(filename, 'w') as f:
        np.savetxt(f, 1/beta_list)
    
    # Run the lanczos averaging
    tic = time.time()
    if isLTLM:
        thermo_dict, thermal_avg_q, emin = compute_thermal_average_LTLM_sector_single(beta_list, dimH, result_evals, result_evecs, result, seeds, U, N_tot)
    else:
        thermo_dict, thermal_avg_q, emin = compute_thermal_average_sector_single_v2(beta_list, dimH, result_evals, result_evecs, result, seeds, U, N_tot)
    toc = time.time()
    print("Computed Lanczos average for sector number %d in %.2f seconds"%(sector_no, toc-tic))

    # save the output
    filename = os.path.join(save_dir, "Nup_%d_Ndown_%d_values.npy"%(N_up, N_down))
    with open(filename, 'wb') as f:
        for qi in range(len(thermal_avg_q)):
            np.save(f, unumpy.nominal_values(thermal_avg_q[qi]))
    filename = os.path.join(save_dir, "Nup_%d_Ndown_%d_errors.npy"%(N_up, N_down))
    with open(filename, 'wb') as f:
        for qi in range(len(thermal_avg_q)):
            np.save(f, unumpy.std_devs(thermal_avg_q[qi]))
    
    thermo_dict_vals = {}
    thermo_dict_errs = {}
    for key in thermo_dict.keys():
        thermo_dict_vals[key] = unumpy.nominal_values(thermo_dict[key])
        thermo_dict_errs[key] = unumpy.std_devs(thermo_dict[key])
    filename = os.path.join(save_dir, "Nup_%d_Ndown_%d_thermo_values.npz"%(N_up, N_down))
    with open(filename, 'wb') as f:
        np.savez(f, **thermo_dict_vals)
    filename = os.path.join(save_dir, "Nup_%d_Ndown_%d_thermo_errors.npz"%(N_up, N_down))
    with open(filename, 'wb') as f:
        np.savez(f, **thermo_dict_errs)
    
    # append emin to file
    filename = os.path.join(save_dir, "Nup_%d_Ndown_%d_emin.txt"%(N_up, N_down))
    with open(filename, 'w') as f:
        f.write(f"{emin}\n")
    
    toc0 = time.time()
    print("Done!")
    return toc0-tic0
#=========================================================================================================================
#
#=========================================================================================================================
def process_sector_single_nematic(geom_folder, int_folder, sector_dict, sector_no, imbalance, isLTLM=False):
    
    tic0 = time.time()
    N_sites, N_up,N_down,dimH = sector_dict[sector_no]
    N_tot = N_up + N_down

    fname_base = "Nup_%d_Ndown_%d_imbalance_%.2f"%(N_up, N_down, imbalance)
    
    U = 0
    result = []
    seeds  = []
    result_evals = []
    result_evecs = []
    keys_to_array = None
    data_dir = os.path.join("..","raw_data",geom_folder,int_folder)
    
    all_files = os.listdir(data_dir)
    print("Found %d files in data_dir folder"%(len(all_files)))
    tic = time.time()
    for filename in all_files:
        if filename.endswith(".npz"):
            pars = filename[:-4].split('_')[1::2]
            nup = int(pars[0])
            ndown = int(pars[1])
            U = float(pars[2])
            imb = float(pars[3])
            seed = int(pars[4]) 
            if nup != N_up or ndown != N_down or imb != imbalance:
                continue
            full_path = os.path.join(data_dir, filename)
            with np.load(full_path) as dfile:
                if keys_to_array is None:
                    keys_to_array = []
                    for k in sorted(dfile.files):
                        if k not in ["evecs", "evals"]:
                            keys_to_array.append(k)
                temp_lst = np.array([dfile[k] for k in keys_to_array])
                result_evals.append(dfile["evals"])
                if seed != 0:
                    result_evecs.append(dfile["evecs"])
                result.append(temp_lst)
                seeds.append(seed)
    toc = time.time()
    print("Loaded %d files of sector number %d in %.2f seconds"%(len(seeds), sector_no, toc-tic))

    save_dir = os.path.join("..", "ED_data",geom_folder,int_folder)
    print(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(save_dir, fname_base+"_quantities_list.txt")
    with open(filename, 'w') as f:
        for key in keys_to_array:
            f.write(f"{key}\n")
    
    if isLTLM:
        beta_list = np.linspace(25, 5, 21)
        beta_list = np.array(list(beta_list) + list(np.linspace(5, 0.01, 11)))
    else:
        beta_list = 1/np.linspace(0.05, 1.05, 21)
        beta_list = np.array(list(beta_list) + list(np.linspace(0.67, 0.01, 11)))
    
    # beta_list = np.linspace(50, 10, 21)
    # beta_list = np.array(list(beta_list) + list(np.linspace(10, 0.1, 11)))

    filename = os.path.join(save_dir, fname_base+"_T_list.txt")
    with open(filename, 'w') as f:
        np.savetxt(f, 1/beta_list)
    
    # Run the lanczos averaging
    tic = time.time()
    if isLTLM:
        thermo_dict, thermal_avg_q, emin = compute_thermal_average_LTLM_sector_single(beta_list, dimH, result_evals, result_evecs, result, seeds, U, N_tot)
    else:
        thermo_dict, thermal_avg_q, emin = compute_thermal_average_sector_single_v2(beta_list, dimH, result_evals, result_evecs, result, seeds, U, N_tot)
    toc = time.time()
    print("Computed Lanczos average for sector number %d in %.2f seconds"%(sector_no, toc-tic))

    # save the output
    filename = os.path.join(save_dir, fname_base+"_values.npy")
    with open(filename, 'wb') as f:
        for qi in range(len(thermal_avg_q)):
            np.save(f, unumpy.nominal_values(thermal_avg_q[qi]))
    filename = os.path.join(save_dir, fname_base+"_errors.npy")
    with open(filename, 'wb') as f:
        for qi in range(len(thermal_avg_q)):
            np.save(f, unumpy.std_devs(thermal_avg_q[qi]))
    
    thermo_dict_vals = {}
    thermo_dict_errs = {}
    for key in thermo_dict.keys():
        thermo_dict_vals[key] = unumpy.nominal_values(thermo_dict[key])
        thermo_dict_errs[key] = unumpy.std_devs(thermo_dict[key])
    filename = os.path.join(save_dir, fname_base+"_thermo_values.npz")
    with open(filename, 'wb') as f:
        np.savez(f, **thermo_dict_vals)
    filename = os.path.join(save_dir, fname_base+"_thermo_errors.npz")
    with open(filename, 'wb') as f:
        np.savez(f, **thermo_dict_errs)
    
    # append emin to file
    filename = os.path.join(save_dir, fname_base+"_emin.txt")
    with open(filename, 'w') as f:
        f.write(f"{emin}\n")
    
    toc0 = time.time()
    print("Done!")
    return toc0-tic0
#=========================================================================================================================
#
#=========================================================================================================================

def process_sector_single_dynamical_response(folder, sector_dict, sector_no, beta, cutoff=None, isLTLM=False):
    
    tic0 = time.time()
    N_sites, N_up,N_down,dimH = sector_dict[sector_no]
    N_tot = N_up + N_down
    print(N_tot, N_up, N_down)
    fname_base = "Nup_%d_Ndown_%d_beta_%.2f"%(N_up, N_down, beta)
    
    U = 0
    seeds  = []
    result_dicts =[]
    data_dir = os.path.join("..","raw_data",folder)
    
    all_files = os.listdir(data_dir)
    print("Found %d files in data_dir folder"%(len(all_files)))
    tic = time.time()
    for filename in all_files:
        if filename.endswith(".npz"):
            pars = filename[:-4].split('_')[1::2]
            nup = int(pars[0])
            ndown = int(pars[1])
            U = float(pars[2])
            seed = int(pars[3]) 
            if nup != N_up or ndown != N_down:
                continue
            full_path = os.path.join(data_dir, filename)
            with open(full_path, 'rb') as dfile:
                data = np.load(dfile, allow_pickle=True)
                result_dicts.append(dict(data))
                seeds.append(seed)
    toc = time.time()
    print("Loaded %d files of sector number %d in %.2f seconds"%(len(seeds), sector_no, toc-tic))

    if cutoff is not None:
        seeds = seeds[:cutoff]
        result_dicts = result_dicts[:cutoff]
    else:
        cutoff=len(seeds)

    save_dir = os.path.join("..", "ED_data",folder)
    print(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if isLTLM:
        pass
    else:
        omega_list = np.linspace(0,16,501)
        t_max = 1/(omega_list[1]-omega_list[0])
        t1 = np.linspace(0,t_max,len(omega_list))

    # Run the lanczos averaging
    tic = time.time()
    if isLTLM:
        pass
    else:
        response_w_real, response_w_imag, response_t_real, response_t_imag, Z, emin = compute_dynamical_response_FTLM_sector_single(omega_list, t1, beta, dimH, result_dicts, seeds, U, N_tot)
    toc = time.time()
    print("Computed Lanczos average for sector number %d in %.2f seconds"%(sector_no, toc-tic))

    savez_dict = dict()
    savez_dict['omega'] = omega_list
    savez_dict['t'] = t1
    savez_dict['Re[C(w)]'] = unumpy.nominal_values(response_w_real)
    savez_dict['Re[C(w)]_err'] = unumpy.std_devs(response_w_real)
    savez_dict['Im[C(w)]'] = unumpy.nominal_values(response_w_imag)
    savez_dict['Im[C(w)]_err'] = unumpy.std_devs(response_w_imag)
    savez_dict['Re[C(t)]'] = unumpy.nominal_values(response_t_real)
    savez_dict['Re[C(t)]_err'] = unumpy.std_devs(response_t_real)
    savez_dict['Im[C(t)]'] = unumpy.nominal_values(response_t_imag)
    savez_dict['Im[C(t)]_err'] = unumpy.std_devs(response_t_imag)
    savez_dict['Z'] = unumpy.nominal_values(Z)
    savez_dict['Z_err'] = unumpy.std_devs(Z)
    savez_dict['emin'] = emin

    filename = os.path.join(save_dir, fname_base+f"_{cutoff}.npz")
    with open(filename, 'wb') as f:
        np.savez(f, **savez_dict)

    # append emin to file
    filename = os.path.join(save_dir, fname_base+"_emin.txt")
    with open(filename, 'w') as f:
        f.write(f"{emin}\n")
    
    toc0 = time.time()
    print("Done!")
    return toc0-tic0


def compute_thermal_average_sector_resolved_v2(beta_list, dimH_dict, result_evals, result_evecs, result, seeds, U):
    '''
    params: 
    beta_list: list of inverse temperatures 
    dimH_dict: dict containing Hilbert space dimensions of the various sectors 
    result_evals, result_evecs, result, seeds
    returns:
    Z_mega_list: dict of partition functions (vs beta) for each sector
    thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    emin_dict: dict of ground state energies for each sector
    '''
    emin_dict, _ = compute_emin_emax_dicts(result_evals, seeds)
    N_list = list(seeds.keys())
    N_quantities = len(result[N_list[0]][0])
    thermal_avg_q = [{} for qi in range(N_quantities)]
    Z_mega_list  = {} # partition function
    F_mega_list = {}  # free energy
    E_mega_list = {}  # average energy
    S_mega_list = {}  # entropy

    for (nu, nd) in N_list:
        Z_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        F_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        E_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        S_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        for qi in range(N_quantities):
            thermal_avg_q[qi][(nu,nd)] = unumpy.uarray(np.zeros(len(beta_list)),np.zeros(len(beta_list)))

    tic_start = time.time()
    for (nu, nd) in tqdm.tqdm(N_list):
        emin = emin_dict[(nu,nd)]
        N_seeds = len(seeds[(nu,nd)])
        if N_seeds>1:
            Z_r = []
            E_r = []
            q_r = [[] for qi in range(N_quantities)]
            #  print(f_i, sztot_i, u_i)
            for r_i in range(N_seeds):
                evals = result_evals[(nu, nd)][r_i]
                evecs = result_evecs[(nu, nd)][r_i]
                evals2 = np.copy(result_evals[(nu, nd)][r_i])
                idx2 = (evals2 != 1e6)
                # evals2[evals2 == 1e6] = U/2*(nu+nd)
#                 Z = np.array([np.sum(np.abs(evecs[0,:])**2 * np.exp(-beta*(evals-emin))) for beta in beta_list])
                Z = np.sum(np.abs(evecs[0,idx2, np.newaxis])**2 * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
                Z_r.append(Z)
                E = np.sum((evals2[idx2,np.newaxis]-U/2*(nu+nd)) * np.abs(evecs[0,idx2, np.newaxis])**2 * \
                                                               np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
                E_r.append(E)

                for qi in range(N_quantities):
                    overlaps_q = result[(nu, nd)][r_i][qi]
#                     overlaps_q = results[qi][f_i][sztot_i][u_i][r_i]
#                     q_temp = np.array([np.sum(evecs[0,:] * np.exp(-beta*(evals-emin))* (np.conjugate(evecs.T) @ overlaps_q)) for beta in beta_list])
                    q_temp = np.sum(evecs[0,idx2, np.newaxis] * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin))* (np.conjugate(evecs.T) @ overlaps_q)[idx2,np.newaxis], axis=0) 
                    q_r[qi].append(q_temp)
                    
            ZR = unumpy.uarray(np.mean(Z_r, axis=0),np.std(Z_r, axis=0)/np.sqrt(N_seeds))
            Z_mega_list[(nu, nd)] = dimH_dict[(nu, nd)] * ZR

            ER = unumpy.uarray(np.mean(E_r, axis=0),np.std(E_r, axis=0)/np.sqrt(N_seeds))
            E_mega_list[(nu, nd)] = dimH_dict[(nu, nd)] * ER / Z_mega_list[(nu, nd)]

            FR_v = -1/beta_list*(np.log(unumpy.nominal_values(Z_mega_list[(nu, nd)])) - beta_list*(emin-U/2*(nu+nd)))
            FR_e = 1/beta_list*unumpy.std_devs(Z_mega_list[(nu, nd)])/unumpy.nominal_values(Z_mega_list[(nu, nd)])
            F_mega_list[(nu, nd)] = unumpy.uarray(FR_v, FR_e)
            S_mega_list[(nu, nd)] = beta_list*(E_mega_list[(nu, nd)]-F_mega_list[(nu, nd)])

            for qi in range(N_quantities):
                qR = unumpy.uarray(np.mean(q_r[qi], axis=0), np.std(q_r[qi], axis=0)/np.sqrt(N_seeds))
                thermal_avg_q[qi][(nu, nd)]  = dimH_dict[(nu, nd)] * qR / Z_mega_list[(nu, nd)]

        else:
            Z_r = np.zeros(len(beta_list))
            E_r = np.zeros(len(beta_list))
            q_r = [np.zeros(len(beta_list)) for qi in range(N_quantities)]
            r_i = 0
            evals = result_evals[(nu,nd)][0]
            # m, beta
            Z_r = np.sum(np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
            Z_mega_list[(nu,nd)] = unumpy.uarray(Z_r, np.zeros_like(Z_r))
            E_r = np.sum((evals[:, np.newaxis]-U/2*(nu+nd)) * np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
            E_mega_list[(nu, nd)] = unumpy.uarray(E_r/Z_r, np.zeros_like(E_r))
            F_mega_list[(nu, nd)] = unumpy.uarray(-1/beta_list*(np.log(Z_r)- beta_list*(emin-U/2*(nu+nd))), np.zeros_like(Z_r))
            S_mega_list[(nu, nd)] = beta_list*(E_mega_list[(nu, nd)]-F_mega_list[(nu, nd)])

            for qi in range(N_quantities):
                overlaps_q = result[(nu,nd)][0][qi]
                q_temp = np.sum(np.exp(-beta_list[np.newaxis,:]*(evals[:, np.newaxis]-emin))* (overlaps_q[:, np.newaxis]), axis=0) 
                q_r[qi] = q_temp / Z_r
                thermal_avg_q[qi][(nu,nd)] = unumpy.uarray(q_r[qi], np.zeros_like(q_r[qi]))
    thermal_avg_q.append(E_mega_list)  # add energy to the list of quantities
    thermo_dict = {'Z':Z_mega_list, 'E':E_mega_list, 'F':F_mega_list, 'S':S_mega_list}
    toc_stop = time.time()
    print(f"Total time taken: {toc_stop-tic_start:.2f} seconds")
    return thermo_dict, thermal_avg_q, emin_dict


def compute_grand_average_v2(mu_list, beta_list, U, scale2, thermo_dict, thermal_avg_q, emin_dict):
    '''
    params:
    mu_list: list of chemical potentials
    beta_list: list of inverse temperatures
    U: interaction strength
    scale2: coefficients used for managing exponential blow up of Z
    Z_mega_list: dict of partition functions (vs beta) for each sector
    thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    emin_dict: dict of ground state energies for each sector

    returns:
    Zmu_list: 2D unumpy.uarray containing grand potential Z(mu, beta)  
    grand_avg_q: list of 2D uarrays containing quantities A(mu, beta) 
    '''
    
    Z_mega_list = thermo_dict['Z']

    N_quantities = len(thermal_avg_q)
    N_list = list(emin_dict.keys())
    N_sites = np.max(N_list)
    
    Zmu_vals = np.zeros((len(mu_list), len(beta_list))) 
    Zmu_variance = np.zeros((len(mu_list), len(beta_list))) 
    Zmu_errs = np.zeros((len(mu_list), len(beta_list))) 
    
    Omega_vals = np.zeros((len(mu_list), len(beta_list))) 
    Omega_errs = np.zeros((len(mu_list), len(beta_list))) 

    N_vals = np.zeros((len(mu_list), len(beta_list))) 
    N_variance = np.zeros((len(mu_list), len(beta_list))) 
    N_errs = np.zeros((len(mu_list), len(beta_list))) 

    grand_avg_q_vals = [np.zeros((len(mu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_variance = [np.zeros((len(mu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_errs = [np.zeros((len(mu_list), len(beta_list))) for qi in range(N_quantities)]

    scale = 1.0
    
    tic_start = time.time()  
    for m_i, mu in enumerate(tqdm.tqdm(mu_list)):
        for (nu,nd) in N_list:
            log_factor =  beta_list*((mu+U/2)*(nu+nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2))
            Zmu_vals[m_i] += unumpy.nominal_values(Z_mega_list[(nu,nd)][:])* np.exp(log_factor)
            Zmu_variance[m_i] += unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2 *np.exp(2*log_factor)
            
        Zmu_errs[m_i] = np.sqrt(Zmu_variance[m_i])
        logZmu = np.log(Zmu_vals[m_i])
        Omega_vals[m_i] = -1/beta_list*logZmu - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2)
        Omega_errs[m_i] = 1/beta_list*Zmu_errs[m_i]/Zmu_vals[m_i]

        for (nu,nd) in N_list:
            log_factor =  beta_list*((mu+U/2)*(nu+nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2))
            ZbetaN = unumpy.nominal_values(Z_mega_list[(nu,nd)][:])
            logZbetaN = np.log(ZbetaN)
            N_vals[m_i,  :] += np.exp(log_factor - logZmu + logZbetaN)*(nu+nd)
            N_variance[m_i, :] +=  np.exp(2*log_factor - 2*logZmu)*\
                                    (nu+nd)**2 * unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
                   
            for q_i in range(N_quantities):
                grand_avg_q_vals[q_i][m_i,  :] += np.exp(log_factor - logZmu + logZbetaN)*\
                                                unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])
                grand_avg_q_variance[q_i][m_i,  :] += np.exp(2*log_factor - 2*logZmu)*\
                                                unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])**2 *\
                                                unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
                
                grand_avg_q_variance[q_i][m_i,  :] += np.exp(2*log_factor - 2*logZmu + 2*logZbetaN)*\
                                                unumpy.std_devs(thermal_avg_q[q_i][(nu,nd)][:])**2
            
                    
        for q_i in range(N_quantities):
            grand_avg_q_variance[q_i][m_i,  :] += Zmu_errs[m_i]**2 *(grand_avg_q_vals[q_i][m_i])**2 * np.exp(- 2*logZmu)
            grand_avg_q_errs[q_i][m_i,  :] = np.sqrt(grand_avg_q_variance[q_i][m_i, :])
        N_variance[m_i, :] += Zmu_errs[m_i]**2 *(N_vals[m_i])**2 * np.exp(- 2*logZmu)
        N_errs[m_i,:] = np.sqrt(N_variance[m_i, :])

    Zmu_list = unumpy.uarray(Zmu_vals, Zmu_errs)
    grand_avg_q = [unumpy.uarray(grand_avg_q_vals[qi], grand_avg_q_errs[qi]) for qi in range(N_quantities)]

    Ebar = grand_avg_q.pop(-1)                     # Average energy
    GP = unumpy.uarray(Omega_vals, Omega_errs)     # Grand Potential (Omega)
    Nbar = unumpy.uarray(N_vals, N_errs)           # Average atom number
    kappa = u_gradient_axis(Nbar, mu_list, 0)      # dN/dmu
    S0 = -u_gradient_axis(GP, 1/beta_list, 1)              # Entropy
    Cv = 1/beta_list*u_gradient_axis(S0, 1/beta_list, 1)   # Heat Capacity (constant volume)
    thermo_dict2 = {'Z':Zmu_list, 'Omega':GP, 'E':Ebar, 'N': Nbar, 'S': S0, 'Cv': Cv, 'kappa': kappa}
    print(f"Total time taken: {time.time()-tic_start:.2f} seconds")
    return thermo_dict2, grand_avg_q



def compute_grand_average_with_Bfield_v2(dmu_list, beta_list, mu, U, scale2, scale3, thermo_dict, thermal_avg_q, emin_dict):
    '''
    params:
    dmu_list: list of chemical potential difference applied to up and down spins
    beta_list: list of inverse temperatures
    mu: chemical potential value to be used 
    U: interaction strength
    scale2, scale3: coefficients used for managing exponential blow up of Z
    Z_mega_list: dict of partition functions (vs beta) for each sector
    thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    emin_dict: dict of ground state energies for each sector

    returns:
    Zdmu_list: 2D unumpy.uarray containing grand potential Z(dmu, beta)  
    grand_avg_q2: list of 2D uarrays containing quantities A(dmu, beta) 
    '''
    Z_mega_list = thermo_dict['Z']

    N_quantities = len(thermal_avg_q)
    N_list = list(emin_dict.keys())
    N_sites = np.max(N_list)
    
    Omega_vals = np.zeros((len(dmu_list), len(beta_list))) 
    Omega_errs = np.zeros((len(dmu_list), len(beta_list))) 

    N_vals = np.zeros((len(dmu_list), len(beta_list))) 
    N_variance = np.zeros((len(dmu_list), len(beta_list))) 
    N_errs = np.zeros((len(dmu_list), len(beta_list))) 

    M_vals = np.zeros((len(dmu_list), len(beta_list))) 
    M_variance = np.zeros((len(dmu_list), len(beta_list))) 
    M_errs = np.zeros((len(dmu_list), len(beta_list))) 

    Zdmu_vals = np.zeros((len(dmu_list), len(beta_list))) 
    Zdmu_variance = np.zeros((len(dmu_list), len(beta_list))) 
    Zdmu_errs = np.zeros((len(dmu_list), len(beta_list))) 

    grand_avg_q_vals2 = [np.zeros((len(dmu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_variance2 = [np.zeros((len(dmu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_errs2 = [np.zeros((len(dmu_list), len(beta_list))) for qi in range(N_quantities)]

    scale = 1.0
    tic_start = time.time()  
    for dm_i, dmu in enumerate(tqdm.tqdm(dmu_list)):
        tic = time.time()    
        for (nu,nd) in N_list:
            log_factor =  beta_list*((mu+dmu/2+U/2)*(nu) + (mu-dmu/2+U/2)*(nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2)- np.exp(abs(dmu) *scale3))
            Zdmu_vals[dm_i] += unumpy.nominal_values(Z_mega_list[(nu,nd)][:])* np.exp(log_factor)
            Zdmu_variance[dm_i] += unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2 *np.exp(2*log_factor)

        Zdmu_errs[dm_i] = np.sqrt(Zdmu_variance[dm_i])
        logZdmu = np.log(Zdmu_vals[dm_i])
        Omega_vals[dm_i] = -1/beta_list*logZdmu - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2) - np.exp(abs(dmu) *scale3)
        Omega_errs[dm_i] = 1/beta_list*Zdmu_errs[dm_i]/Zdmu_vals[dm_i]

        for (nu,nd) in N_list:
            log_factor =  beta_list*((mu+dmu/2+U/2)*(nu) + (mu-dmu/2+U/2)*(nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2) - np.exp(abs(dmu) * scale3))
            ZbetaN = unumpy.nominal_values(Z_mega_list[(nu,nd)][:])
            logZbetaN = np.log(ZbetaN)
            N_vals[dm_i,  :] += np.exp(log_factor - logZdmu + logZbetaN)*(nu+nd)
            N_variance[dm_i, :] +=  np.exp(2*log_factor - 2*logZdmu)*\
                                    (nu+nd)**2 * unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
            M_vals[dm_i,  :] += np.exp(log_factor - logZdmu + logZbetaN)*(nu-nd)
            M_variance[dm_i, :] +=  np.exp(2*log_factor - 2*logZdmu)*\
                                    (nu-nd)**2 * unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
            
            for q_i in range(N_quantities):
                grand_avg_q_vals2[q_i][dm_i,  :] += np.exp(log_factor - logZdmu + logZbetaN)*\
                                                unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])
                grand_avg_q_variance2[q_i][dm_i,  :] += np.exp(2*log_factor - 2*logZdmu)*\
                                                unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])**2 *\
                                                unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
                grand_avg_q_variance2[q_i][dm_i,  :] += np.exp(2*log_factor - 2*logZdmu + 2*logZbetaN)*\
                                                unumpy.std_devs(thermal_avg_q[q_i][(nu,nd)][:])**2
                
        for q_i in range(N_quantities):
            grand_avg_q_variance2[q_i][dm_i,  :] += Zdmu_errs[dm_i]**2 *(grand_avg_q_vals2[q_i][dm_i])**2 * np.exp(- 2*logZdmu)
            grand_avg_q_errs2[q_i][dm_i,  :] = np.sqrt(grand_avg_q_variance2[q_i][dm_i, :])
        N_variance[dm_i, :] += Zdmu_errs[dm_i]**2 *(N_vals[dm_i])**2 * np.exp(- 2*logZdmu)
        N_errs[dm_i,:] = np.sqrt(N_variance[dm_i, :])
        M_variance[dm_i, :] += Zdmu_errs[dm_i]**2 *(N_vals[dm_i])**2 * np.exp(- 2*logZdmu)
        M_errs[dm_i,:] = np.sqrt(M_variance[dm_i, :])


    Zdmu_list = unumpy.uarray(Zdmu_vals, Zdmu_errs)
    grand_avg_q2 = [unumpy.uarray(grand_avg_q_vals2[qi], grand_avg_q_errs2[qi]) for qi in  range(N_quantities)]
    Ebar = grand_avg_q2.pop(-1)                     # Average energy
    GP = unumpy.uarray(Omega_vals, Omega_errs)      # Grand Potential (Omega)
    Nbar = unumpy.uarray(N_vals, N_errs)            # Average atom number
    Mbar = unumpy.uarray(M_vals, N_errs)            # Average atom number
    chi = u_gradient_axis(Mbar, 0.5*dmu_list, 0)    # dM/dh (h=0.5*dmu)
    S0 = -u_gradient_axis(GP, 1/beta_list, 1)              # Entropy
    Cv = 1/beta_list*u_gradient_axis(S0, 1/beta_list, 1)   # Heat Capacity (constant volume)
    thermo_dict2 = {'Z':Zdmu_list, 'Omega':GP, 'E':Ebar, 'N': Nbar, 'S': S0, 'Cv': Cv, 'M':Mbar, 'chi': chi}
    print(f"Total time taken: {time.time()-tic_start:.2f} seconds")
    return thermo_dict2, grand_avg_q2



def compute_grand_average_with_mu_B(mu_list2, dmu_list, beta_list, U, scale2, scale3, Z_mega_list, thermal_avg_q, emin_dict):
    '''
    params:
    mu_list2: list of chemical potentials
    dmu_list: list of chemical potential difference applied to up and down spins
    beta_list: list of inverse temperatures
    U: interaction strength
    scale2, scale3: coefficients used for managing exponential blow up of Z
    Z_mega_list: dict of partition functions (vs beta) for each sector
    thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    emin_dict: dict of ground state energies for each sector

    returns:
    Zmudmu_list: 3D unumpy.uarray containing grand potential Z(mu, dmu, beta)  
    grand_avg_q3: list of 3D uarrays containing quantities A(mu, dmu, beta) 
    '''

    N_quantities = len(thermal_avg_q)
    N_list = list(emin_dict.keys())
    N_sites = np.max(N_list)
    
    Zmudmu_vals = np.zeros((len(mu_list2), len(dmu_list), len(beta_list))) 
    Zmudmu_variance = np.zeros((len(mu_list2), len(dmu_list), len(beta_list))) 
    Zmudmu_errs = np.zeros((len(mu_list2), len(dmu_list), len(beta_list))) 
    grand_avg_q_vals3 = [np.zeros((len(mu_list2), len(dmu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_variance3 = [np.zeros((len(mu_list2), len(dmu_list), len(beta_list))) for qi in range(N_quantities)]
    grand_avg_q_errs3 = [np.zeros((len(mu_list2), len(dmu_list), len(beta_list))) for qi in range(N_quantities)]

    scale = 1.0
    tic_start = time.time()
    for m_i, mu in enumerate(tqdm.tqdm(mu_list2)):  
        for dm_i, dmu in enumerate(dmu_list):
            for (nu,nd) in N_list:
                log_factor =  beta_list*((mu+dmu/2+U/2)*(nu) + (mu-dmu/2+U/2)*(nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2)- np.exp(abs(dmu) *scale3))
                Zmudmu_vals[m_i,dm_i] += unumpy.nominal_values(Z_mega_list[(nu,nd)][:])* np.exp(log_factor)
                Zmudmu_variance[m_i,dm_i] += unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2 *np.exp(2*log_factor)
            Zmudmu_errs[m_i,dm_i] = np.sqrt(Zmudmu_variance[m_i,dm_i])
            logZmudmu = np.log(Zmudmu_vals[m_i,dm_i])
            for (nu,nd) in N_list:
                log_factor =  beta_list*((mu+dmu/2+U/2)*(nu) + (mu-dmu/2+U/2)*(nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2) - np.exp(abs(dmu) * scale3))
                ZbetaN = unumpy.nominal_values(Z_mega_list[(nu,nd)][:])
                logZbetaN = np.log(ZbetaN)
                for q_i in range(N_quantities):
                    grand_avg_q_vals3[q_i][m_i,dm_i,  :] += np.exp(log_factor - logZmudmu + logZbetaN)*\
                                                    unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])
                    grand_avg_q_variance3[q_i][m_i,dm_i,  :] += np.exp(2*log_factor - 2*logZmudmu)*\
                                                    unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])**2 *\
                                                    unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
                    grand_avg_q_variance3[q_i][m_i,dm_i,  :] += np.exp(2*log_factor - 2*logZmudmu + 2*logZbetaN)*\
                                                    unumpy.std_devs(thermal_avg_q[q_i][(nu,nd)][:])**2
                    
            for q_i in range(N_quantities):
                grand_avg_q_variance3[q_i][m_i,dm_i,  :] += Zmudmu_errs[m_i,dm_i]**2 *(grand_avg_q_vals3[q_i][m_i,dm_i])**2 * np.exp(- 2*logZmudmu)
                grand_avg_q_errs3[q_i][m_i,dm_i,  :] = np.sqrt(grand_avg_q_variance3[q_i][m_i,dm_i, :])
        
    Zmudmu_list = unumpy.uarray(Zmudmu_vals, Zmudmu_errs)
    grand_avg_q3 = [unumpy.uarray(grand_avg_q_vals3[qi], grand_avg_q_errs3[qi]) for qi in  range(N_quantities)]
    print(f"Total time taken: {time.time()-tic_start:.2f} seconds")
    return Zmudmu_list, grand_avg_q3


# Helper function
def u_gradient(uy, ux):
    '''
    Equivalent of uy = np.gradient(uy, ux) but for unumpy
    computes dy/dx
    '''
    dx = ux[1:]-ux[:-1]
    dx1 = dx[0:-1]
    dx2 = dx[1:]
    slice1 = slice(1, -1)
    slice2 = slice(None, -2)
    slice3 = slice(1, -1)
    slice4 = slice(2, None)
    a = -(dx2)/(dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))
    out = unumpy.uarray(np.zeros(len(uy)), np.zeros(len(uy)))
    out[slice1] = a * uy[slice2] + b * uy[slice3] + c * uy[slice4]

    slice1 = 0
    slice2 = 1
    slice3 = 0
    dx_0 = dx[0]
    out[slice1] = (uy[slice2] - uy[slice3]) / dx_0

    slice1 = -1
    slice2 = -1
    slice3 = -2
    dx_n = dx[-1]
    out[slice1] = (uy[slice2] - uy[slice3]) / dx_n
    return out

# Helper function
def u_gradient_axis(uy, ux, axis=0):
    '''
    Equivalent of uy = np.gradient(uy, ux) but for unumpy
    computes dy/dx
    computes gradient only along single axis (default = 0)
    '''
    N = len(np.shape(uy))
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    out = unumpy.uarray(np.zeros(np.shape(uy)), np.zeros(np.shape(uy)))
    
    # Numerical differentiation: 2nd order interior
    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(None, -2)
    slice3[axis] = slice(1, -1)
    slice4[axis] = slice(2, None)

    dx = ux[1:]-ux[:-1]
    dx1 = dx[0:-1]
    dx2 = dx[1:]
    a = -(dx2)/(dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))
    shape = np.ones(N, dtype=int)
    shape[axis] = -1
    a.shape=b.shape=c.shape=shape

    out[tuple(slice1)] = a * uy[tuple(slice2)] + b * uy[tuple(slice3)] + c * uy[tuple(slice4)]

    # Numerical differentiation: 1st order edges
    slice1[axis] = 0
    slice2[axis] = 1
    slice3[axis] = 0
    dx_0 = dx[0]
    out[tuple(slice1)] = (uy[tuple(slice2)] - uy[tuple(slice3)]) / dx_0

    slice1[axis] = -1
    slice2[axis] = -1
    slice3[axis] = -2
    dx_n = dx[-1]
    out[tuple(slice1)] = (uy[tuple(slice2)] - uy[tuple(slice3)]) / dx_n

    return out



def compute_thermal_average_LTLM_sector_resolved_v2(beta_list, dimH_dict, result_evals, result_evecs, result, seeds, U):
    '''
    params: 
    beta_list: list of inverse temperatures 
    dimH_dict: dict containing Hilbert space dimensions of the various sectors 
    result_evals, result_evecs, result, seeds
    returns:
    Z_mega_list: dict of partition functions (vs beta) for each sector
    thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    emin_dict: dict of ground state energies for each sector
    '''
    emin_dict, _ = compute_emin_emax_dicts(result_evals, seeds)
    N_list = list(seeds.keys())
    N_quantities = len(result[N_list[0]][0])
    thermal_avg_q = [{} for qi in range(N_quantities)]
    Z_mega_list  = {} # partition function
    F_mega_list = {}  # free energy
    E_mega_list = {}  # average energy
    S_mega_list = {}  # entropy

    for (nu, nd) in N_list:
        Z_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        F_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        E_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        S_mega_list[(nu,nd)]   = unumpy.uarray(np.zeros(len(beta_list)), np.zeros(len(beta_list)))
        for qi in range(N_quantities):
            thermal_avg_q[qi][(nu,nd)] = unumpy.uarray(np.zeros(len(beta_list)),np.zeros(len(beta_list)))

    tic_start = time.time()
    for (nu, nd) in tqdm.tqdm(N_list):
        emin = emin_dict[(nu,nd)]
        N_seeds = len(seeds[(nu,nd)])
        if N_seeds>1:
            Z_r = []
            E_r = []
            q_r = [[] for qi in range(N_quantities)]
            #  print(f_i, sztot_i, u_i)
            for r_i in range(N_seeds):
                evals = result_evals[(nu, nd)][r_i]
                evecs = result_evecs[(nu, nd)][r_i]
                evals2 = np.copy(result_evals[(nu, nd)][r_i])
                idx2 = (evals2 != 1e6)
                Z = np.sum(np.abs(evecs[0,idx2, np.newaxis])**2 * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
                Z_r.append(Z)
                E = np.sum((evals2[idx2,np.newaxis]-U/2*(nu+nd)) * np.abs(evecs[0,idx2, np.newaxis])**2 * \
                                                               np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin)), axis=0)
                E_r.append(E)

                for qi in range(N_quantities):
                    overlaps_q = result[(nu, nd)][r_i][qi]
                    # q_temp = np.sum(evecs[0,idx2, np.newaxis] * np.exp(-beta_list[np.newaxis, :]*(evals[idx2,np.newaxis]-emin))* (np.conjugate(evecs.T) @ overlaps_q)[idx2,np.newaxis], axis=0) 
                    q_temp = np.einsum('bjl,j,l,mn,nl,mj->b', 
                                        np.exp(-beta_list[np.newaxis, np.newaxis, :]*(evals[idx2, np.newaxis, np.newaxis]-emin + evals[np.newaxis, idx2, np.newaxis]-emin)/2),
                                        evecs[0,idx2],
                                        np.conjugate(evecs[0,idx2]),
                                        overlaps_q,
                                        evecs[:,idx2],
                                        np.conjugate(evecs[:,idx2].T)
                                        )
                    q_r[qi].append(q_temp)
                    
            ZR = unumpy.uarray(np.mean(Z_r, axis=0),np.std(Z_r, axis=0)/np.sqrt(N_seeds))
            Z_mega_list[(nu, nd)] = dimH_dict[(nu, nd)] * ZR

            ER = unumpy.uarray(np.mean(E_r, axis=0),np.std(E_r, axis=0)/np.sqrt(N_seeds))
            E_mega_list[(nu, nd)] = dimH_dict[(nu, nd)] * ER / Z_mega_list[(nu, nd)]

            FR_v = -1/beta_list*(np.log(unumpy.nominal_values(Z_mega_list[(nu, nd)])) - beta_list*(emin-U/2*(nu+nd)))
            FR_e = 1/beta_list*unumpy.std_devs(Z_mega_list[(nu, nd)])/unumpy.nominal_values(Z_mega_list[(nu, nd)])
            F_mega_list[(nu, nd)] = unumpy.uarray(FR_v, FR_e)
            S_mega_list[(nu, nd)] = beta_list*(E_mega_list[(nu, nd)]-F_mega_list[(nu, nd)])

            for qi in range(N_quantities):
                qR = unumpy.uarray(np.mean(q_r[qi], axis=0), np.std(q_r[qi], axis=0)/np.sqrt(N_seeds))
                thermal_avg_q[qi][(nu, nd)]  = dimH_dict[(nu, nd)] * qR / Z_mega_list[(nu, nd)]

        else:
            Z_r = np.zeros(len(beta_list))
            E_r = np.zeros(len(beta_list))
            q_r = [np.zeros(len(beta_list)) for qi in range(N_quantities)]
            r_i = 0
            evals = result_evals[(nu,nd)][0]
            # m, beta
            Z_r = np.sum(np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
            Z_mega_list[(nu,nd)] = unumpy.uarray(Z_r, np.zeros_like(Z_r))
            E_r = np.sum((evals[:, np.newaxis]-U/2*(nu+nd)) * np.exp(-beta_list[np.newaxis, :]*(evals[:, np.newaxis]-emin)), axis=0)
            E_mega_list[(nu, nd)] = unumpy.uarray(E_r/Z_r, np.zeros_like(E_r))
            F_mega_list[(nu, nd)] = unumpy.uarray(-1/beta_list*(np.log(Z_r)- beta_list*(emin-U/2*(nu+nd))), np.zeros_like(Z_r))
            S_mega_list[(nu, nd)] = beta_list*(E_mega_list[(nu, nd)]-F_mega_list[(nu, nd)])

            for qi in range(N_quantities):
                overlaps_q = result[(nu,nd)][0][qi]
                q_temp = np.sum(np.exp(-beta_list[np.newaxis,:]*(evals[:, np.newaxis]-emin))* (overlaps_q[:, np.newaxis]), axis=0) 
                q_r[qi] = q_temp / Z_r
                thermal_avg_q[qi][(nu,nd)] = unumpy.uarray(q_r[qi], np.zeros_like(q_r[qi]))
    thermal_avg_q.append(E_mega_list)  # add energy to the list of quantities
    thermo_dict = {'Z':Z_mega_list, 'E':E_mega_list, 'F':F_mega_list, 'S':S_mega_list}
    toc_stop = time.time()
    print(f"Total time taken: {toc_stop-tic_start:.2f} seconds")
    return thermo_dict, thermal_avg_q, emin_dict



#def compute_grand_average_FTLM_dynamical_response_v2(mu_list, omega_list, beta, U, scale2, thermo_dict, thermal_avg_q, emin_dict):
    # '''
    # params:
    # mu_list: list of chemical potentials
    # beta_list: list of inverse temperatures
    # U: interaction strength
    # scale2: coefficients used for managing exponential blow up of Z
    # Z_mega_list: dict of partition functions (vs beta) for each sector
    # thermal_avg_q: list of dicts for thermal avgs of the various quantities in each sector
    # emin_dict: dict of ground state energies for each sector

    # returns:
    # Zmu_list: 2D unumpy.uarray containing grand potential Z(mu, beta)  
    # grand_avg_q: list of 2D uarrays containing quantities A(mu, beta) 
    # '''
    
    # Z_mega_list = thermo_dict['Z']

    # N_quantities = len(thermal_avg_q)
    # N_list = list(emin_dict.keys())
    # N_sites = np.max(N_list)
    
    # Zmu_vals = np.zeros((len(mu_list), len(omega_list))) 
    # Zmu_variance = np.zeros((len(mu_list), len(omega_list))) 
    # Zmu_errs = np.zeros((len(mu_list), len(omega_list))) 
    
    # Omega_vals = np.zeros((len(mu_list), len(omega_list))) 
    # Omega_errs = np.zeros((len(mu_list), len(omega_list))) 

    # N_vals = np.zeros((len(mu_list), len(omega_list))) 
    # N_variance = np.zeros((len(mu_list), len(omega_list))) 
    # N_errs = np.zeros((len(mu_list), len(omega_list))) 

    # grand_avg_q_vals = [np.zeros((len(mu_list), len(omega_list))) for qi in range(N_quantities)]
    # grand_avg_q_variance = [np.zeros((len(mu_list), len(omega_list))) for qi in range(N_quantities)]
    # grand_avg_q_errs = [np.zeros((len(mu_list), len(omega_list))) for qi in range(N_quantities)]

    # scale = 1.0
    
    # tic_start = time.time()  
    # for m_i, mu in enumerate(tqdm.tqdm(mu_list)):
    #     for (nu,nd) in N_list:
    #         log_factor =  omega_list*((mu+U/2)*(nu+nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2))
    #         Zmu_vals[m_i] += unumpy.nominal_values(Z_mega_list[(nu,nd)][:])* np.exp(log_factor)
    #         Zmu_variance[m_i] += unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2 *np.exp(2*log_factor)
            
    #     Zmu_errs[m_i] = np.sqrt(Zmu_variance[m_i])
    #     logZmu = np.log(Zmu_vals[m_i])
    #     Omega_vals[m_i] = -1/omega_list*logZmu - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2)
    #     Omega_errs[m_i] = 1/omega_list*Zmu_errs[m_i]/Zmu_vals[m_i]

    #     for (nu,nd) in N_list:
    #         log_factor =  omega_list*((mu+U/2)*(nu+nd) - emin_dict[(nu,nd)] - (mu+U/2)*scale*N_sites - np.exp(abs(mu) * scale2))
    #         ZbetaN = unumpy.nominal_values(Z_mega_list[(nu,nd)][:])
    #         logZbetaN = np.log(ZbetaN)
    #         N_vals[m_i,  :] += np.exp(log_factor - logZmu + logZbetaN)*(nu+nd)
    #         N_variance[m_i, :] +=  np.exp(2*log_factor - 2*logZmu)*\
    #                                 (nu+nd)**2 * unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
                   
    #         for q_i in range(N_quantities):
    #             grand_avg_q_vals[q_i][m_i,  :] += np.exp(log_factor - logZmu + logZbetaN)*\
    #                                             unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])
    #             grand_avg_q_variance[q_i][m_i,  :] += np.exp(2*log_factor - 2*logZmu)*\
    #                                             unumpy.nominal_values(thermal_avg_q[q_i][(nu,nd)][:])**2 *\
    #                                             unumpy.std_devs(Z_mega_list[(nu,nd)][:])**2
                
    #             grand_avg_q_variance[q_i][m_i,  :] += np.exp(2*log_factor - 2*logZmu + 2*logZbetaN)*\
    #                                             unumpy.std_devs(thermal_avg_q[q_i][(nu,nd)][:])**2
            
                    
    #     for q_i in range(N_quantities):
    #         grand_avg_q_variance[q_i][m_i,  :] += Zmu_errs[m_i]**2 *(grand_avg_q_vals[q_i][m_i])**2 * np.exp(- 2*logZmu)
    #         grand_avg_q_errs[q_i][m_i,  :] = np.sqrt(grand_avg_q_variance[q_i][m_i, :])
    #     N_variance[m_i, :] += Zmu_errs[m_i]**2 *(N_vals[m_i])**2 * np.exp(- 2*logZmu)
    #     N_errs[m_i,:] = np.sqrt(N_variance[m_i, :])

    # Zmu_list = unumpy.uarray(Zmu_vals, Zmu_errs)
    # grand_avg_q = [unumpy.uarray(grand_avg_q_vals[qi], grand_avg_q_errs[qi]) for qi in range(N_quantities)]

    # Ebar = grand_avg_q.pop(-1)                     # Average energy
    # GP = unumpy.uarray(Omega_vals, Omega_errs)     # Grand Potential (Omega)
    # Nbar = unumpy.uarray(N_vals, N_errs)           # Average atom number
    # kappa = u_gradient_axis(Nbar, mu_list, 0)      # dN/dmu
    # S0 = -u_gradient_axis(GP, 1/omega_list, 1)              # Entropy
    # Cv = 1/omega_list*u_gradient_axis(S0, 1/omega_list, 1)   # Heat Capacity (constant volume)
    # thermo_dict2 = {'Z':Zmu_list, 'Omega':GP, 'E':Ebar, 'N': Nbar, 'S': S0, 'Cv': Cv, 'kappa': kappa}
    # print(f"Total time taken: {time.time()-tic_start:.2f} seconds")
    # return thermo_dict2, grand_avg_q
