"""Solve the sabra shell model

Example
-------
python sabra_model/sabra_model.py --time_to_run=1
"""
import sys

sys.path.append("..")
from math import ceil
import numpy as np
from numba import njit, types
from pyinstrument import Profiler
from shell_model_experiments.sabra_model.runge_kutta4 import runge_kutta4
from shell_model_experiments.params.params import *
import shell_model_experiments.utils.util_funcs as sh_utils
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
import general.utils.argument_parsers as a_parsers
import config as cfg

profiler = Profiler()

# Set global params
cfg.GLOBAL_PARAMS.record_max_time = 30


@njit(
    (
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 2, "C", readonly=False),
        types.int64,
        types.float64,
        types.float64,
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def run_model(
    u_old: np.ndarray,
    du_array: np.ndarray,
    data_out: np.ndarray,
    Nt_local: int,
    ny: float,
    forcing: float,
    diff_exponent: int,
):
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

        # Solve nonlinear terms + forcing
        u_old = runge_kutta4(
            y0=u_old,
            h=dt,
            du=du_array,
            forcing=forcing,
        )
        # Solve linear diffusive term explicitly
        u_old[bd_size:-bd_size] = u_old[bd_size:-bd_size] * np.exp(
            -ny * k_vec_temp ** diff_exponent * dt
        )

    return u_old


def main(args=None):

    # Define u_old
    u_old = (u0 * initial_k_vec).astype(np.complex128)
    u_old = np.pad(u_old, pad_width=bd_size, mode="constant")

    # Get number of records
    args["n_records"] = ceil((args["Nt"]) / int(cfg.GLOBAL_PARAMS.record_max_time / dt))

    # Write ref info file
    g_save.save_reference_info(args)

    profiler.start()
    print(
        f'\nRunning sabra model for {args["Nt"]*dt:.2f}s with a burn-in time'
        + f' of {args["burn_in_time"]:.2f}s, i.e. {args["n_records"]:d} records '
        + f"are saved to disk each with {cfg.GLOBAL_PARAMS.record_max_time:.1f}s data\n"
    )

    # Burn in the model for the desired burn in time
    data_out = np.zeros(
        (int(args["burn_in_time"] * tts), sdim + 1), dtype=np.complex128
    )

    print(f'running burn-in phase of {args["burn_in_time"]}s\n')
    u_old = run_model(
        u_old,
        du_array,
        data_out,
        int(args["burn_in_time"] / dt),
        args["ny"],
        args["forcing"],
        args["diff_exponent"],
    )

    for ir in range(args["n_records"]):
        # Calculate data out size
        # Set standard size
        out_array_size = int(cfg.GLOBAL_PARAMS.record_max_time * tts)
        # If last record -> adjust if size should not equal default record length
        if ir == (args["n_records"] - 1):
            if args["Nt"] % int(cfg.GLOBAL_PARAMS.record_max_time / dt) > 0:
                out_array_size = int(
                    (args["Nt"] % int(cfg.GLOBAL_PARAMS.record_max_time / dt))
                    * sample_rate
                )

        data_out = np.zeros((out_array_size, sdim + 1), dtype=np.complex128)

        # Run model
        print(f'running record {ir + 1}/{args["n_records"]}')
        u_old = run_model(
            u_old,
            du_array,
            data_out,
            int(out_array_size / sample_rate),
            args["ny"],
            args["forcing"],
            args["diff_exponent"],
        )

        # Add record_id to datafile header
        args["record_id"] = ir
        if not args["skip_save_data"]:
            print(f"saving record\n")
            save_path = g_save.save_data(data_out, args=args)

    if args["erda_run"]:
        stand_data_name = g_save_utils.generate_standard_data_name(args)
        compress_out_name = f"ref_data_{stand_data_name}"
        g_save_utils.compress_dir(save_path, compress_out_name)

    profiler.stop()
    print(profiler.output_text())


if __name__ == "__main__":
    # Get arguments
    stand_arg_setup = a_parsers.StandardRunnerArgSetup()
    stand_arg_setup.setup_parser()
    args = stand_arg_setup.args

    args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
        args["forcing"], args["ny_n"], args["diff_exponent"]
    )

    args["Nt"] = int(args["time_to_run"] / dt)

    main(args=args)
