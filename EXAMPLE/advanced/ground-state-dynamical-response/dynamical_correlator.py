import sys
import numpy as np
from scipy.special import comb
from methods.geometry import *
from pathlib import Path
import os
import heating_helper
import random

# show_plots = True
show_plots = False

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

Tvec12 = np.array([3,0])
Tvec34 = np.array([0,3])
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


def make_geometry_x(boundary):
    
    cvecs = [ce1, -ce1]

    if boundary == "PBC":
        connectivity_list = generate_connectivity_list(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    else:
        connectivity_list = generate_connectivity_list_OBC(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    
    
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
    return connectivity_list


def make_geometry_y(boundary):
    cvecs = [ce2, -ce2]
    if boundary == "PBC":
        connectivity_list = generate_connectivity_list(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    else:
        connectivity_list = generate_connectivity_list_OBC(sites, sdict, lines, Tvecs, cvecs, e1, e2)
    
    
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
    return connectivity_list


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
            #   (measurements_FH.create_doublon_func_i,(4,)),
            #   (measurements_FH.create_density_func_i,(4,)),
            #   (measurements_FH.create_Sz_func_i,(4,)),
            #   (measurements_FH.create_Sz_func_i,(5,)),
            #   (measurements_FH.create_Sz_func_i,(2,)),
            #   (measurements_FH.create_singles_func,()),
            #   (measurements_FH.create_SzSz_func,(NN1_bonds+NN2_bonds,)),
            #   (measurements_FH.create_SzSz_func,(NN1_bonds,)),
            #   (measurements_FH.create_SzSz_func,(NN1_bonds,)),
            #   (measurements_FH.create_SzSz_func,(NN2_bonds,)),   
            #   (measurements_FH.create_pp_func,(NN1_bonds,)),
            #   (measurements_FH.create_pp_func,(NN2_bonds,)),   
            #   (measurements_FH.create_nn_func,(NN1_bonds,)),
            #   (measurements_FH.create_nn_func,(NN2_bonds,)),   
    ]
    return quantities

if __name__== '__main__':
    bc = (sys.argv[1])
    connectivity_list, folder_name, geom_dict = make_geometry(bc)
    conx = make_geometry_x(bc)
    cony = make_geometry_y(bc)
    
    N_sites = len(connectivity_list)

    filling  = np.arange(1, N_sites+1)
    
    N_up_lists = [np.arange(max(0,f-min(N_sites,f)), min(N_sites+1,f+1)) for f in filling]
    N_down_lists = [np.arange(min(N_sites,f),max(-1,f-min(N_sites,f)-1),-1) for f in filling]
    Sz_tot_lists = [nup_list - ndown_list for (nup_list, ndown_list) in zip(N_up_lists,N_down_lists)]
    dimH_lists = [[int(comb(N_sites, nu)*comb(N_sites, nd)) for (nu,nd) in zip(N_ups, N_downs)] for (N_ups, N_downs) in zip(N_up_lists, N_down_lists)]
    
    U = float(sys.argv[2])
    N_states = 250

    modulation_type = sys.argv[3]

    direc = os.path.join("raw_data", "heating_response_"+modulation_type, folder_name,"U%d"%U)
    print(direc)
    Path(direc).mkdir(parents=True, exist_ok=True)

    seed_start = random.randint(1e6,1e7)
    print("Seed start integer = ", seed_start)

    # run_particular_symmetry_sector(N_sites, N_up, N_down, dimH, U, modulation,  connectivity_list, conx, cony, direc, N_states=100, seed_start=1000)

    argslst = []
    sectors_lst = []
    for f_i, f in enumerate(filling):
        for sztot_i, (sztot, N_up, N_down) in enumerate(zip(Sz_tot_lists[f_i], N_up_lists[f_i], N_down_lists[f_i])):
            if N_up >= N_down:
                dimH = dimH_lists[f_i][sztot_i]
                argslst.append((N_sites, N_up, N_down, dimH, U, modulation_type, connectivity_list, conx, cony, direc, N_states, seed_start))
                sectors_lst.append((N_sites, N_up, N_down, dimH))
    argslst.sort(key=lambda x:x[3])
    sectors_lst.sort(key=lambda x:x[3])
    
    print(len(argslst))
    with open(f"sectors_{N_sites}sites.txt", 'w') as f:
        f.writelines(str(i) + ' ' + str(line) + '\n' for i,line in enumerate(sectors_lst))
    
    if len(sys.argv) == 5:
        idx = int(sys.argv[4])
        print(idx, sectors_lst[idx])#, argslst[idx])
        elapsed_time = heating_helper.run_particular_symmetry_sector(*argslst[idx])
        print("Time taken for sector ({},{},{}): {} = {:.2f} seconds".format(*argslst[idx][:4], elapsed_time))
    else:
        for sysarg in sys.argv[4:]:
            idx = int(sysarg)
            print(idx, sectors_lst[idx])
            elapsed_time = heating_helper.run_particular_symmetry_sector(*argslst[idx])
            print("Time taken for sector ({},{},{}): {} = {:.2f} seconds".format(*argslst[idx][:4], elapsed_time))