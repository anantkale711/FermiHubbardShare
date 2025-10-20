import sys
import numpy as np
from scipy.special import comb
from methods.geometry import *
from pathlib import Path
import os
from leo import measurements_FH
import time_evolution_helper
import random

show_plots = False

if show_plots:
    import matplotlib.pyplot as plt

d2r = np.pi/180.
e1 = np.array([1,0])
e2 = np.array([0,1])

ce1 = np.array([1,0])
ce2 = np.array([0,1])
# ce3 = np.array([1,1])
cvecs = [ce1, -ce1, ce2, -ce2]

# Tvec12 = np.array([-2,-3])
# Tvec34 = np.array([4,-2])
Tvec12 = np.array([3,0])
Tvec34 = np.array([0,3])

def make_geometry(boundary):
    

    Tvecs = [Tvec12, -Tvec12, Tvec34, -Tvec34]

    lines = generate_line_functions(Tvecs, e1, e2, flip=False)
    count, sites, sdict = generate_lattice_sites(e1, e2, lines)
    print(count)
    N_sites= count
    
    if boundary == "PBC":
        connectivity_list = generate_connectivity_list(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    else:
        connectivity_list = generate_connectivity_list_OBC(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    
    folder_name = "Square_%d_(%d,%d)_(%d,%d)_%s"%(N_sites, Tvecs[0][0], Tvecs[0][1], Tvecs[2][0], Tvecs[2][1], boundary)
    
    geom_dict = {'e1':e1, 'e2':e2, 'cvecs':cvecs, 'Tvecs': Tvecs, 'lines':lines, 'sites':sites, 'sdict': sdict}

    if show_plots:
        print(connectivity_list)
        # print('Here')
        fig, ax = plt.subplots()
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
        plt.xlim(-3,6)
        plt.ylim(-6,2)
        # plt.xlim(-0.5,4)
        # plt.ylim(-4.5,1.5)

        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()
    return connectivity_list, folder_name, geom_dict


def make_geometry_x(boundary):
    

    Tvecs = [Tvec12, -Tvec12, Tvec34, -Tvec34]

    lines = generate_line_functions(Tvecs, e1, e2, flip=False)
    count, sites, sdict = generate_lattice_sites(e1, e2, lines)
    print(count)
    N_sites= count
    
    if boundary == "PBC":
        connectivity_list = generate_connectivity_list(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    else:
        connectivity_list = generate_connectivity_list_OBC(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    
    
    if show_plots:
        print(connectivity_list)
        # print('Here')
        fig, ax = plt.subplots()
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
        plt.xlim(-3,6)
        plt.ylim(-6,2)
        # plt.xlim(-0.5,4)
        # plt.ylim(-4.5,1.5)

        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()
    return connectivity_list


def make_geometry_y(boundary):

    Tvecs = [Tvec12, -Tvec12, Tvec34, -Tvec34]

    lines = generate_line_functions(Tvecs, e1, e2, flip=False)
    count, sites, sdict = generate_lattice_sites(e1, e2, lines)
    print(count)
    N_sites= count
    
    if boundary == "PBC":
        connectivity_list = generate_connectivity_list(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    else:
        connectivity_list = generate_connectivity_list_OBC(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    
    
    if show_plots:
        print(connectivity_list)
        # print('Here')
        fig, ax = plt.subplots()
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
        plt.xlim(-3,6)
        plt.ylim(-6,2)
        # plt.xlim(-0.5,4)
        # plt.ylim(-4.5,1.5)

        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()
    return connectivity_list


def make_measurements_list(geom_dict):
    
    cvecs = geom_dict['cvecs']
    ce1 = cvecs[0]
    ce2 = cvecs[2]
    cvecsNN = cvecs
    cvecsNNNc = [ce1+ce2, ce1-ce2, -ce1-ce2, ce1+ce2]
    cvecsNNNs = [2*ce for ce in cvecs]
    cvecsNNNNc = [ce1+2*ce2, ce2+2*ce1, ce1-2*ce2, ce2-2*ce1, -ce1-2*ce2, -ce2-2*ce1, -ce1+2*ce2, -ce2+2*ce1]

    NN1_bonds = generate_bonds_list_PBC(geom_dict['sites'], geom_dict['sdict'], geom_dict['lines'], geom_dict['Tvecs'], 
                                       cvecs[:2], 
                                       geom_dict['e1'], geom_dict['e2'])
    
    NN2_bonds = generate_bonds_list_PBC(geom_dict['sites'], geom_dict['sdict'], geom_dict['lines'], geom_dict['Tvecs'], 
                                       cvecs[2:], 
                                       geom_dict['e1'], geom_dict['e2'])
    
    quantities = [
              (measurements_FH.create_SzSz_func,(NN1_bonds,)),
              (measurements_FH.create_SzSz_func,(NN2_bonds,)),   
              (measurements_FH.create_pp_func,(NN1_bonds,)),
              (measurements_FH.create_pp_func,(NN2_bonds,)),   
              (measurements_FH.create_nn_func,(NN1_bonds,)),
              (measurements_FH.create_nn_func,(NN2_bonds,)),   
    ]
    return quantities

if __name__== '__main__':
    
    connectivity_list, folder_name, geom_dict = make_geometry("PBC")
    conx = make_geometry_x("PBC")
    cony = make_geometry_y("PBC")
    
    N_sites = len(connectivity_list)

    filling  = np.arange(1, N_sites+1)
    
    N_up_lists = [np.arange(max(0,f-min(N_sites,f)), min(N_sites+1,f+1)) for f in filling]
    N_down_lists = [np.arange(min(N_sites,f),max(-1,f-min(N_sites,f)-1),-1) for f in filling]
    Sz_tot_lists = [nup_list - ndown_list for (nup_list, ndown_list) in zip(N_up_lists,N_down_lists)]
    dimH_lists = [[int(comb(N_sites, nu)*comb(N_sites, nd)) for (nu,nd) in zip(N_ups, N_downs)] for (N_ups, N_downs) in zip(N_up_lists, N_down_lists)]
    
    U = 8.0
    N_states = 150

    direc = os.path.join("raw_data", "time_evolution", folder_name,"U%d"%U)
    print(direc)
    Path(direc).mkdir(parents=True, exist_ok=True)

    measurement_list = make_measurements_list(geom_dict)

    # seed_start = int(sys.argv[2])
    seed_start = random.randint(1e6,1e7)
    print("Seed start integer = ", seed_start)
    ramp_duration = float(sys.argv[1])

    imbalance_initial = float(sys.argv[2])
    imbalance_final = float(sys.argv[3])

    argslst = []
    sectors_lst = []
    for f_i, f in enumerate(filling):
        for sztot_i, (sztot, N_up, N_down) in enumerate(zip(Sz_tot_lists[f_i], N_up_lists[f_i], N_down_lists[f_i])):
            if N_up >= N_down:
                dimH = dimH_lists[f_i][sztot_i]
                argslst.append((N_sites, U, N_up, N_down, measurement_list, ramp_duration, dimH, connectivity_list, conx, cony, direc, imbalance_initial, imbalance_final, N_states, seed_start))
                sectors_lst.append((N_sites, N_up, N_down, dimH))
    argslst.sort(key=lambda x:x[6])
    sectors_lst.sort(key=lambda x:x[3])
    
    print(len(argslst))
    with open(f"sectors_{N_sites}sites_PBC.txt", 'w') as f:
        f.writelines(str(i) + ' ' + str(line) + '\n' for i,line in enumerate(sectors_lst))
    
    if len(sys.argv) == 5:
        idx = int(sys.argv[4])
        elapsed_time = time_evolution_helper.run_particular_symmetry_sector(*argslst[idx])
        print("Time taken for sector ({},{},{}): {} = {:.2f} seconds".format(N_sites, argslst[idx][2], argslst[idx][3], argslst[idx][6], elapsed_time))
    else:
        for sysarg in sys.argv[4:]:
            idx = int(sysarg)
            elapsed_time = time_evolution_helper.run_particular_symmetry_sector(*argslst[idx])
            print("Time taken for sector ({},{},{}): {} = {:.2f} seconds".format(N_sites, argslst[idx][2], argslst[idx][3], argslst[idx][6], elapsed_time))