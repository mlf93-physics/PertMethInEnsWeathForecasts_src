import os
import sys
sys.path.append('..')
from math import floor, log10
import argparse
from pathlib import Path
import numpy as np
from numba import jit, types
import multiprocessing
from pyinstrument import Profiler
import matplotlib.pyplot as plt
from src.sabra_model.sabra_model import run_model
from src.params.params import *
from src.utils.save_data_funcs import save_data, save_perturb_info
from src.utils.import_data_funcs import import_header, import_ref_data,\
    import_start_u_profiles
from src.utils.dev_plots import dev_plot_eigen_mode_analysis,\
    dev_plot_perturbation_generation
from src.utils.util_funcs import match_start_positions_to_ref_file,\
    get_sorted_ref_record_names

profiler = Profiler()
# @jit((types.Array(types.complex128, 2, 'C', readonly=True),
#        types.Array(types.complex128, 2, 'C', readonly=False),
#        types.Array(types.complex128, 2, 'C', readonly=False),
#        types.boolean, types.int64, types.float64), parallel=True, cache=True)
def find_eigenvector_for_perturbation(u_init_profiles,
        dev_plot_active=False, n_profiles=None, local_ny=None):
    """Find the eigenvector corresponding to the minimal of the positive
    eigenvalues of the initial vel. profile.
    
    Use the form of the sabra model to perform the calculation of the Jacobian
    matrix. Perform singular-value-decomposition to get the eigenvalues and
    -vectors. Choose the minimal of the positive eigenvalues with respect to
    the real part of the eigenvalue.

    Parameters
    ----------
    u_init_profiles : ndarray
        The initial velocity profiles

    Returns
    -------
    max_e_vector : ndarray
        The eigenvectors corresponding to the minimal of the positive eigenvalues
    
    """
    print('\nFinding the eigenvalues and eigenvectors at the position of the' +
        ' given velocity profiles\n')

    # Prepare for returning all eigen vectors and values
    e_vector_collection = []
    e_value_collection = []

    e_vector_matrix = np.zeros((n_k_vec, n_profiles), dtype=np.complex128)

    # Perform the conjugation
    u_init_profiles_conj = u_init_profiles.conj()
    # Prepare prefactor vector to multiply on J_matrix
    prefactor_reshaped = np.reshape(pre_factor, (-1, 1))
    # Perform calculation for all u_profiles
    for i in range(n_profiles):
        # Calculate the Jacobian matrix
        J_matrix = np.zeros((n_k_vec, n_k_vec), dtype=np.complex128)
        # Add k=2 diagonal
        J_matrix += np.diag(
            u_init_profiles_conj[bd_size+1:-bd_size - 1, i], k=2)
        # Add k=1 diagonal
        J_matrix += factor2*np.diag(np.concatenate((np.array([0 + 0j]),
            u_init_profiles_conj[bd_size:-bd_size - 2, i])), k=1)
        # Add k=-1 diagonal
        J_matrix += factor3*np.diag(np.concatenate((np.array([0 + 0j]),
            u_init_profiles[bd_size:-bd_size - 2, i])), k=-1)
        # Add k=-2 diagonal
        J_matrix += factor3*np.diag(
            u_init_profiles[bd_size+1:-bd_size - 1, i], k=-2)


        # Add contribution from derivatives of the complex conjugates:
        J_matrix += np.diag(np.concatenate((u_init_profiles[bd_size + 2:-bd_size, i], np.array([0 + 0j]))), k=1)
        J_matrix += factor2*np.diag(np.concatenate((u_init_profiles[bd_size + 2:-bd_size, i], np.array([0 + 0j]))), k=-1)

        J_matrix = J_matrix*prefactor_reshaped

        # Add the k=0 diagonal
        # temp_ny = args['ny'] if header is None else header['ny']
        J_matrix -= np.diag(local_ny * k_vec_temp**2, k=0)

        e_values, e_vectors = np.linalg.eig(J_matrix)
        
        e_vector_collection.append(e_vectors)
        e_value_collection.append(e_values)

        # positive_e_values_indices = np.argwhere(e_values.real > 0)
        chosen_e_value_index =\
            np.argmax(e_values.real)

        e_vector_matrix[:, i] = e_vectors[:, chosen_e_value_index]
        J_matrix.fill(0 + 0j)
    
        # if dev_plot_active:
        #     print('Largest positive eigenvalue', e_values[chosen_e_value_index])
            
        #     dev_plot_eigen_mode_analysis(e_values, J_matrix, e_vectors,
        #         header=header, perturb_pos=perturb_positions[i])

    return e_vector_matrix, e_vector_collection, e_value_collection


def calculate_perturbations(perturb_e_vectors, dev_plot_active=False,
        args=None):
    """Calculate a random perturbation with a specific norm for each profile.
    
    The norm of the error is defined in the parameter seeked_error_norm

    Parameters
    ----------
    perturb_e_vectors : ndarray
        The eigenvectors along which to perform the perturbations

    Returns
    -------
    perturbations : ndarray
        The random perturbations
    
    """
    n_profiles = args['n_profiles']
    n_runs_per_profile = args['n_runs_per_profile']
    perturbations = np.zeros((n_k_vec + 2*bd_size, n_profiles*
        n_runs_per_profile), dtype=np.complex128)

    # Perform perturbation for all eigenvectors
    for i in range(n_profiles*n_runs_per_profile):
        # Generate random error
        error = np.random.rand(2*n_k_vec).astype(np.float64)*2 - 1
        # Apply single shell perturbation
        if args['single_shell_perturb'] is not None:
            perturb = np.zeros(n_k_vec, dtype=np.complex128)
            perturb.real[args['single_shell_perturb']] = error[0]
            perturb.imag[args['single_shell_perturb']] = error[1]
        else:
            # Reshape into complex array
            perturb = np.empty(n_k_vec, dtype=np.complex128)
            perturb.real = error[:n_k_vec]
            perturb.imag = error[n_k_vec:]
        # Copy array for plotting
        perturb_temp = np.copy(perturb)

        # Scale random perturbation with the normalised eigenvector
        perturb = perturb*perturb_e_vectors[:, i // n_runs_per_profile]
        # Find scaling factor in order to have the seeked norm of the error
        lambda_factor = seeked_error_norm/np.linalg.norm(perturb)
        # Scale down the perturbation
        perturb = lambda_factor*perturb

        # Perform small test to be noticed if the perturbation is not as expected
        np.testing.assert_almost_equal(np.linalg.norm(perturb), seeked_error_norm,
            decimal=abs(floor(log10(seeked_error_norm))) + 1)

        perturbations[bd_size:-bd_size, i] = perturb

        if dev_plot_active:
            dev_plot_perturbation_generation(perturb, perturb_temp)

    return perturbations

def perturbation_runner(u_old, perturb_positions, du_array, data_out, args,
    run_count, perturb_count):
    """Execute the sabra model on one given perturbed u_old profile"""

    print(f'Running perturbation {run_count + 1}/' + 
        f"{args['n_profiles']*args['n_runs_per_profile']} | profile" +
        f" {run_count // args['n_runs_per_profile']}, profile run" +
        f" {run_count % args['n_runs_per_profile']}")
    
    run_model(u_old, du_array, data_out, args['Nt'], args['ny'], args['forcing'])
    save_data(data_out, subfolder=Path(args['path']).name, prefix=f'perturb{perturb_count}_',
        perturb_position=perturb_positions[run_count // args['n_runs_per_profile']],
        args=args)

def main(args=None):
    args['Nt'] = int(args['time_to_run']/dt)
    args['burn_in_lines'] = int(args['burn_in_time']/dt*sample_rate)

    # Import start profiles
    u_init_profiles, perturb_positions, header_dict = import_start_u_profiles(
        args=args)
     
    # Save parameters to args dict:
    args['forcing'] = header_dict['f'].real
    
    if args['ny_n'] is None:
        args['ny'] = header_dict['ny']

        if args['forcing'] == 0:
            args['ny_n'] = 0
        else:    
            args['ny_n'] = int(3/8*log10(args['forcing']/(header_dict['ny']**2))/log10(lambda_const))
        # Take ny from reference file
    else:
        args['ny'] = (args['forcing']/(lambda_const**(8/3*args['ny_n'])))**(1/2)
    
    print('args', args)


    if args['eigen_perturb']:
        print('\nRunning with eigen_perturb\n')
        perturb_e_vectors, _, _ = find_eigenvector_for_perturbation(
            u_init_profiles[:, 0:args['n_profiles']*
                args['n_runs_per_profile']:args['n_runs_per_profile']],
            dev_plot_active=False, n_profiles=args['n_profiles'],
            local_ny=header_dict['ny'])
    else:
        print('\nRunning without eigen_perturb\n')
        perturb_e_vectors = np.ones((n_k_vec, args['n_profiles']),
            dtype=np.complex128)

    if args['single_shell_perturb'] is not None:
        print('\nRunning in single shell perturb mode\n')

    # Make perturbations
    perturbations = calculate_perturbations(perturb_e_vectors,
        dev_plot_active=False, args=args)

    # Prepare array for saving
    data_out = np.zeros((int(args['Nt']*sample_rate), n_k_vec + 1), dtype=np.complex128)

    # Detect if other perturbations exist in the perturbation_folder and calculate
    # perturbation count to start at
    # Check if path exists
    expected_path = Path(args['path'], args['perturb_folder'])
    dir_exists = os.path.isdir(expected_path)
    if dir_exists:
        n_perturbation_files = len(list(expected_path.glob('*.csv')))
    else:
        n_perturbation_files = 0

    # Prepare and start the perturbation_runner in multiple processes
    processes = []
    profiler.start()

    # Get number of threads
    cpu_count = multiprocessing.cpu_count()

    # Append processes
    for j in range(args['n_runs_per_profile']*args['n_profiles']//cpu_count):
        for i in range(cpu_count):
            count = j*cpu_count + i

            processes.append(multiprocessing.Process(target=perturbation_runner,
                args=(u_init_profiles[:, count] + perturbations[:, count], perturb_positions,
                    du_array, data_out, args, count, count + n_perturbation_files)))
            processes[-1].start()

        for i in range(len(processes)):
            processes[i].join()
        
        processes = []
    
    for i in range(args['n_runs_per_profile']*args['n_profiles'] % cpu_count):
        count = (args['n_runs_per_profile']*args['n_profiles']//cpu_count)*cpu_count + i

        processes.append(multiprocessing.Process(target=perturbation_runner,
                args=(u_init_profiles[:, count] + perturbations[:, count], perturb_positions,
                    du_array, data_out, args, count, count + n_perturbation_files)))
        processes[-1].start()


    for i in range(len(processes)):
        processes[i].join()

    profiler.stop()
    print(profiler.output_text())


    save_perturb_info(args=args)
    

if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", nargs='+', type=str)
    arg_parser.add_argument("--path", nargs='?', type=str)
    arg_parser.add_argument("--perturb_folder", nargs='?', default=None,
        required=True, type=str)
    arg_parser.add_argument("--time_to_run", default=0.1, type=float)
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--ny_n", default=None, type=int)
    arg_parser.add_argument("--n_runs_per_profile", default=1, type=int)
    arg_parser.add_argument("--n_profiles", default=1, type=int)
    arg_parser.add_argument("--start_time", nargs='+', type=float)
    arg_parser.add_argument("--eigen_perturb", action='store_true')
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    arg_parser.add_argument("--single_shell_perturb", default=None, type=int)
    arg_parser.add_argument("--start_time_offset", default=None, type=float)

    args = vars(arg_parser.parse_args())

    args['ref_run'] = False


    if args['start_time'] is not None:
        if args['n_profiles'] > 1 and args['start_time_offset'] is None:
            np.testing.assert_equal(len(args['start_time']), args['n_profiles'],
                'The number of start times do not equal the number of' +
                ' requested profiles.')
        elif args['n_profiles'] > 1 and args['start_time_offset'] is not None:
            np.testing.assert_equal(len(args['start_time']), 1,
                'Too many start times given')
            print('Determining starttimes from single starttime value and the'+
                ' start_time_offset parameter')
            args['start_time'] = [args['start_time'][0] +
                args['start_time_offset']*i for i in range(args['n_profiles'])]
        else:
            np.testing.assert_equal(len(args['start_time']), 1,
                'Too many start times given')


    # Set seed if wished
    if args['seed_mode']:
        np.random.seed(seed=1)

    main(args=args)