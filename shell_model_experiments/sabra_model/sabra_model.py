import sys
sys.path.append('..')
import argparse
from math import ceil
import numpy as np
from numba import njit, types
from pyinstrument import Profiler
from src.sabra_model.runge_kutta4 import runge_kutta4_vec
from src.params.params import *
from src.utils.save_data_funcs import save_data

profiler = Profiler()

@njit((types.Array(types.complex128, 1, 'C', readonly=False),
       types.Array(types.complex128, 1, 'C', readonly=False),
       types.Array(types.complex128, 2, 'C', readonly=False),
       types.int64, types.float64, types.float64), cache=True)
def run_model(u_old, du_array, data_out, Nt_local, ny, forcing):
    """Execute the integration of the sabra shell model.
    
    Parameters
    ----------
    u_old : ndarray
        The initial shell velocity profile
    du_array : ndarray
        A helper array used to store the current derivative of the shell
        velocities.
    data_out : ndarray
        An array to store samples of the integrated shell velocities.
    
    """
    sample_number = 0
    # Perform calculations
    for i in range(Nt_local):
        # Save samples for plotting
        if i % int(1/sample_rate) == 0:
            data_out[sample_number, 0] = dt*i + 0j
            data_out[sample_number, 1:] = u_old[bd_size:-bd_size]
            sample_number += 1
        
        # Update u_old
        u_old = runge_kutta4_vec(y0=u_old, h=dt, du=du_array, ny=ny, forcing=forcing)
    
    return u_old

def main(args=None):
    
    # Define u_old
    u_old = (u0*initial_k_vec).astype(np.complex128)
    u_old = np.pad(u_old, pad_width=bd_size, mode='constant')

    # Get number of records
    args['n_records'] = ceil((args['Nt'] - args['burn_in_time']/dt) /\
        int(args["record_max_time"]/dt))

    profiler.start()
    print(f'\nRunning sabra model for {args["Nt"]*dt:.2f}s with a burn-in time' +
        f' of {args["burn_in_time"]:.2f}s, i.e. {args["n_records"]:d} records '+
        f'are saved to disk each with {args["record_max_time"]:.1f}s data\n')
    
    # Burn in the model for the desired burn in time
    data_out = np.zeros((int(args["burn_in_time"]*sample_rate/dt), n_k_vec + 1),
            dtype=np.complex128)
    print(f'running burn-in phase of {args["burn_in_time"]}s\n')
    u_old = run_model(u_old, du_array, data_out, int(args['burn_in_time']/dt),
            args['ny'], args['forcing'])

    for ir in range(args['n_records']):
        # Calculate data out size
        if ir == (args['n_records'] - 1):
            if (args['Nt'] - args['burn_in_time']/dt) %\
                    int(args["record_max_time"]/dt) > 0:
                out_array_size = int((args['Nt'] - args['burn_in_time']/dt % \
                    int(args["record_max_time"]/dt))*sample_rate)
        else:
            out_array_size = int(args["record_max_time"]*sample_rate/dt)

        data_out = np.zeros((out_array_size, n_k_vec + 1),
            dtype=np.complex128)

        # Run model
        print(f'running record {ir + 1}/{args["n_records"]}')
        u_old = run_model(u_old, du_array, data_out, out_array_size/sample_rate,
            args['ny'], args['forcing'])

        # Add record_id to datafile header
        args['record_id'] = ir
        print(f'saving record\n')
        save_data(data_out, args=args)

    profiler.stop()
    print(profiler.output_text())


if __name__ == "__main__": 
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--ny_n", default=19, type=int)
    arg_parser.add_argument("--forcing", default=1, type=float)
    arg_parser.add_argument("--save_folder", nargs='?', default='data', type=str)
    arg_parser.add_argument("--record_max_time", default=30, type=float)
    time_group = arg_parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument("--time_to_run", type=float)
    time_group.add_argument("--n_turnovers", type=float)

    args = vars(arg_parser.parse_args())

    args['ny'] = (args['forcing']/(lambda_const**(8/3*args['ny_n'])))**(1/2) #1e-8
    args['ref_run'] = True

    if args['n_turnovers'] is not None:
        args['time_to_run'] = ceil(eddy0_turnover_time*args['n_turnovers'])   # [s]
        args['Nt'] = int(args['time_to_run']/dt)
    else:
        args['Nt'] = int(args['time_to_run']/dt)

    main(args=args)
