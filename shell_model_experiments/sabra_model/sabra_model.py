"""Solve the sabra shell model

Example
-------
python sabra_model/sabra_model.py --time_to_run=1
"""
import sys

sys.path.append("..")
from math import ceil

import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
import general.utils.user_interface as g_ui
import numba as nb
import numpy as np
import shell_model_experiments.utils.util_funcs as ut_funcs
from pyinstrument import Profiler
from shell_model_experiments.params.params import PAR, ParamsStructType
import shell_model_experiments.utils.special_params as sparams
from shell_model_experiments.sabra_model.runge_kutta4 import runge_kutta4
import shell_model_experiments.utils.custom_decorators as dec


profiler = Profiler()

# Set global params
cfg.GLOBAL_PARAMS.record_max_time = 30


@dec.diffusion_type_decorator
@nb.njit(
    (nb.types.Array(nb.types.complex128, 1, "C", readonly=False))(
        nb.types.Array(nb.types.complex128, 1, "C", readonly=False),
        nb.types.Array(nb.types.complex128, 2, "C", readonly=False),
        nb.types.int64,
        nb.types.float64,
        nb.types.float64,
        nb.types.float64,
        nb.typeof(PAR),
    ),
    cache=cfg.NUMBA_CACHE,
)
def run_model(
    diffusion_func: callable,
    u_old: np.ndarray,
    data_out: np.ndarray,
    Nt_local: int,
    ny: float,
    forcing: float,
    diff_exponent: int,
    PAR,
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
        if i % int(1 / PAR.sample_rate) == 0:
            data_out[sample_number, 0] = PAR.dt * i + 0j
            data_out[sample_number, 1:] = u_old[PAR.bd_size : -PAR.bd_size]
            sample_number += 1

        # Solve nonlinear terms + forcing
        u_old = runge_kutta4(y0=u_old, forcing=forcing, PAR=PAR)

        # Solve diffusion depending on method
        u_old = diffusion_func(u_old, ny, diff_exponent)

    return u_old


def main(args=None):

    # Define u_old
    u_old = (PAR.u0 * PAR.initial_k_vec).astype(np.complex128)
    u_old = np.pad(u_old, pad_width=PAR.bd_size, mode="constant")

    # Get number of records
    args["n_records"] = ceil(
        (args["Nt"]) / int(cfg.GLOBAL_PARAMS.record_max_time / PAR.dt)
    )

    # Write ref info file
    g_save.save_reference_info(args)

    profiler.start()
    print(
        f'\nRunning sabra model for {args["Nt"]*PAR.dt:.2f}s with a burn-in time'
        + f' of {args["burn_in_time"]:.2f}s, i.e. {args["n_records"]:d} records '
        + f"are saved to disk each with {cfg.GLOBAL_PARAMS.record_max_time:.1f}s data\n"
    )

    if args["burn_in_time"] > 0:
        # Burn in the model for the desired burn in time
        data_out = np.zeros(
            (int(args["burn_in_time"] * PAR.tts), PAR.sdim + 1), dtype=sparams.dtype
        )

        print(f'running burn-in phase of {args["burn_in_time"]}s\n')
        u_old = run_model(
            u_old,
            data_out,
            int(args["burn_in_time"] / PAR.dt),
            args["ny"],
            args["forcing"],
            args["diff_exponent"],
            PAR,
            diff_type=args["diff_type"],
        )

    for ir in range(args["n_records"]):
        # Calculate data out size
        # Set standard size
        out_array_size = int(cfg.GLOBAL_PARAMS.record_max_time * PAR.tts)
        # If last record -> adjust if size should not equal default record length
        if ir == (args["n_records"] - 1):
            if args["Nt"] % int(cfg.GLOBAL_PARAMS.record_max_time / PAR.dt) > 0:
                out_array_size = int(
                    (args["Nt"] % int(cfg.GLOBAL_PARAMS.record_max_time / PAR.dt))
                    * PAR.sample_rate
                )

        data_out = np.zeros((out_array_size, PAR.sdim + 1), dtype=sparams.dtype)

        # Run model
        print(f'running record {ir + 1}/{args["n_records"]}')
        u_old = run_model(
            u_old,
            data_out,
            int(out_array_size / PAR.sample_rate),
            args["ny"],
            args["forcing"],
            args["diff_exponent"],
            PAR,
            diff_type=args["diff_type"],
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
    print(profiler.output_text(color=True))


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_arg_setup = a_parsers.StandardRunnerArgSetup()
    stand_arg_setup.setup_parser()
    args = stand_arg_setup.args

    g_ui.confirm_run_setup(args)

    # Initiate and update variables and arrays
    ut_funcs.update_dependent_params(PAR, sdim=int(args["sdim"]))
    ut_funcs.update_arrays(PAR)
    args["ny"] = ut_funcs.ny_from_ny_n_and_forcing(
        args["forcing"], args["ny_n"], args["diff_exponent"]
    )

    args["Nt"] = int(args["time_to_run"] / PAR.dt)

    main(args=args)
