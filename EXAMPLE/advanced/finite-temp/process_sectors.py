import sys
import numpy as np
import time
import os
from pathlib import Path
import methods.Lanczos_ensemble_average as LEA
import multiprocessing


def get_sector_dict_from_file(sector_file):
    sector_dict = {}
    with open(sector_file, 'r') as f:
        for line in f:
            # sector_no (N_sites, N_up, N_down, dimH)
            line_table = [s.strip('(,)\n ') for s in line.split(' ')]
            # print(line_table)
            sector_no = int(line_table[0])
            N_sites = int(line_table[1]) 
            N_up = int(line_table[2])
            N_down = int(line_table[3])
            dimH = int(line_table[4])
            sector_dict[sector_no] = (N_sites, N_up, N_down, dimH)
            # print(sector_dict[sector_no])
    return sector_dict


if __name__ == '__main__':
    raw_data_folder = sys.argv[1]
    save_folder = sys.argv[2]
    sector_file = sys.argv[3]

    sector_dict = get_sector_dict_from_file(sector_file)

    if len(sys.argv) == 5:
        sector = int(sys.argv[4])
        print("Sector {}, {}".format(sector, sector_dict[sector]))
        elapsed_time = LEA.process_sector_single(raw_data_folder, save_folder, sector_dict, sector)
        print('Total time for ',sector, ' is: {:.2f} seconds'.format(elapsed_time))
    else:
        if len(sys.argv) == 4:
            sectors = list(sector_dict.keys())
        else:
            sectors = [int(sys.argv[i]) for i in range(4,len(sys.argv)+1)]
        max_processes = 8
        args_list = []
        for sector in sectors:
            args_list.append((raw_data_folder,save_folder, sector_dict, sector))
        print("starting multithreaded processing with %d sectors"%len(args_list))
        p = multiprocessing.Pool(processes = max_processes)
        output = p.starmap(LEA.process_sector_single, args_list)   
        elapsed_time = np.sum(output)
        print('Total time for ',sectors, ' is: {:.2f} seconds'.format(elapsed_time))

    
    