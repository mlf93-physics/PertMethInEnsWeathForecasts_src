import sys

sys.path.append("..")
from math import ceil
import numpy as np
from numba import njit, types
from pyinstrument import Profiler
from shell_model_experiments.sabra_model.runge_kutta4 import runge_kutta4_vec
from shell_model_experiments.params.params import *
import general.utils.saving.save_data_funcs as g_save
import general.utils.argument_parsers as a_parsers
from config import NUMBA_CACHE, GLOBAL_PARAMS

profiler = Profiler()

# Set global params
GLOBAL_PARAMS.record_max_time = 30


@njit(
    (
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 2, "C", readonly=False),
        types.int64,
        types.float64,
        types.float64,
    ),
    cache=NUMBA_CACHE,
)
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
        if i % int(1 / sample_rate) == 0:
            data_out[sample_number, 0] = dt * i + 0j
            data_out[sample_number, 1:] = u_old[bd_size:-bd_size]
            sample_number += 1

        # Update u_old
        u_old = runge_kutta4_vec(y0=u_old, h=dt, du=du_array, ny=ny, forcing=forcing)

    return u_old


def main(args=None):

    # Define u_old
    u_old = (u0 * initial_k_vec).astype(np.complex128)
    u_old = np.pad(u_old, pad_width=bd_size, mode="constant")

    # Get number of records
    args["n_records"] = ceil(
        (args["Nt"] - args["burn_in_time"] / dt)
        / int(GLOBAL_PARAMS.record_max_time / dt)
    )

    profiler.start()
    print(
        f'\nRunning sabra model for {args["Nt"]*dt:.2f}s with a burn-in time'
        + f' of {args["burn_in_time"]:.2f}s, i.e. {args["n_records"]:d} records '
        + f"are saved to disk each with {GLOBAL_PARAMS.record_max_time:.1f}s data\n"
    )

    # Burn in the model for the desired burn in time
    data_out = np.zeros(
        (int(args["burn_in_time"] * tts), n_k_vec + 1), dtype=np.complex128
    )
    print(f'running burn-in phase of {args["burn_in_time"]}s\n')
    u_old = run_model(
        u_old,
        du_array,
        data_out,
        int(args["burn_in_time"] / dt),
        args["ny"],
        args["forcing"],
    )

    for ir in range(args["n_records"]):
        # Calculate data out size
        if ir == (args["n_records"] - 1):
            if (args["Nt"] - args["burn_in_time"] / dt) % int(
                GLOBAL_PARAMS.record_max_time / dt
            ) > 0:
                out_array_size = int(
                    (
                        args["Nt"]
                        - args["burn_in_time"]
                        / dt
                        % int(GLOBAL_PARAMS.record_max_time / dt)
                    )
                    * sample_rate
                )
        else:
            out_array_size = int(GLOBAL_PARAMS.record_max_time * tts)

        data_out = np.zeros((out_array_size, n_k_vec + 1), dtype=np.complex128)

        # Run model
        print(f'running record {ir + 1}/{args["n_records"]}')
        u_old = run_model(
            u_old,
            du_array,
            data_out,
            out_array_size / sample_rate,
            args["ny"],
            args["forcing"],
        )

        # Add record_id to datafile header
        args["record_id"] = ir
        if not args["skip_save_data"]:
            print(f"saving record\n")
            g_save.save_data(data_out, args=args)

    profiler.stop()
    print(profiler.output_text())


if __name__ == "__main__":
    # Get arguments
    stand_arg_setup = a_parsers.StandardRunnerArgSetup()
    stand_arg_setup.setup_parser()
    args = vars(stand_arg_setup.args)

    args["ny"] = (args["forcing"] / (lambda_const ** (8 / 3 * args["ny_n"]))) ** (1 / 2)

    args["Nt"] = int(args["time_to_run"] / dt)

    main(args=args)
