import os
import numpy as np
from pathlib import Path
from src.params.params import *

def save_data(data_out, subfolder="", prefix="", perturb_position=None,
    args=None):
    """Save the data to disc."""

    if args is None:
        print('Please supply an argument dictionary to save data.')
        exit()
    
    
    # Prepare variables to be used when saving
    n_data = data_out.shape[0]
    temp_time_to_run = "{:.2e}".format(args['time_to_run'])
    temp_forcing = "{:.1f}".format(args['forcing'])
    temp_ny = "{:.2e}".format(args['ny'])

    if len(subfolder) == 0:
        expected_name = f"ny{temp_ny}_t{temp_time_to_run}" +\
            f"_n_f{n_forcing}_f{temp_forcing}"
        expected_path = f'./data/{expected_name}'
        
        # See if folder is present
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.mkdir(expected_path)
            
        subfolder = expected_name
    else:
        # Check if path exists
        expected_path = f'./data/{subfolder}'
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.mkdir(expected_path)



    if args['ref_run']:
        ref_header_extra = f", rec_id={args['record_id']}"
        subsubfolder = 'ref_data'
        # Check if path exists
        expected_path = f'./data/{subfolder}/{subsubfolder}'
        dir_exists = os.path.isdir(expected_path)

        # Make dir if not present
        if not dir_exists:
            os.mkdir(expected_path)

        prefix = "ref_"

        ref_filename_extra = f"_rec{args['record_id']}"

        ref_data_info_name = f"data/{subfolder}/{subsubfolder}/ref_data_info_ny"+\
            f"{temp_ny}_t{temp_time_to_run}"+\
            f"_n_f{n_forcing}_f{temp_forcing}.txt"
        info_line = f"f={args['forcing']}, n_f={n_forcing}, n_ny={args['ny_n']}, " +\
             f"ny={args['ny']}, time={args['time_to_run']}, dt={dt}, epsilon={epsilon}, " +\
             f"lambda={lambda_const}, n_records={args['n_records']}, " +\
             f"burn_in_time={args['burn_in_time']}, " +\
             f"record_max_time={args['record_max_time']}, " +\
             f"sample_rate={sample_rate}"
        with open(ref_data_info_name, 'w') as file:
            file.write(info_line)

    else:
        ref_header_extra = ""
        ref_filename_extra = ""
        subsubfolder = args['perturb_folder']

        # Check if path exists
        expected_path = f'./data/{subfolder}/{subsubfolder}'
        dir_exists = os.path.isdir(expected_path)

        # Make dir if not present
        if not dir_exists:
            os.mkdir(expected_path)
    
    header = f"f={args['forcing']}, n_f={n_forcing}, n_ny={args['ny_n']}, ny={args['ny']}, " +\
             f"time={args['time_to_run']}, dt={dt}, epsilon={epsilon}, " +\
             f"lambda={lambda_const}, N_data={n_data}, " +\
             f"sample_rate={sample_rate}" + ref_header_extra
    
    if perturb_position is not None:
        header += f', perturb_pos={int(perturb_position)}'

    # Save data
    np.savetxt(f"data/{subfolder}/{subsubfolder}/{prefix}udata_ny{temp_ny}_t{temp_time_to_run}"+
        f"_n_f{n_forcing}_f{temp_forcing}{ref_filename_extra}.csv",
        data_out, delimiter=",", header=header)
    
def save_perturb_info(args=None):
    """Save info textfile about the perturbation runs"""

    temp_time_to_run = "{:.2e}".format(args['time_to_run'])
    temp_forcing = "{:.1f}".format(args['forcing'])
    temp_ny = "{:.2e}".format(args['ny'])

    # Prepare filename
    perturb_data_info_name = Path(args['path'], args['perturb_folder'], 
        f"perturb_data_info_ny{temp_ny}_t{temp_time_to_run}"+\
        f"_n_f{n_forcing}_f{temp_forcing}.txt")

    # Check if path already exists
    dir_exists = os.path.isdir(perturb_data_info_name)
    if dir_exists:
        return
    
    print('Saving perturb data info textfile\n')

    # Prepare line to write
    info_line = f"f={args['forcing']}, n_f={n_forcing}, n_ny={args['ny_n']}, " +\
            f"ny={args['ny']}, time={args['time_to_run']}, dt={dt}, epsilon={epsilon}, " +\
            f"lambda={lambda_const}, " +\
            f"burn_in_time={args['burn_in_time']}, " +\
            f"sample_rate={sample_rate}, eigen_perturb={args['eigen_perturb']}, "+\
            f"seed_mode={args['seed_mode']}, "+\
            f"single_shell_perturb={args['single_shell_perturb']}, "+\
            f"start_time_offset={args['start_time_offset']}"
        
    # Write to file
    with open(str(perturb_data_info_name), 'w') as file:
        file.write(info_line)
