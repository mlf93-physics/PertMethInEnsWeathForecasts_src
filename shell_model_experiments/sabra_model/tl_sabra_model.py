"""Solve the TL sabra shell model

Example
-------
python sabra_model/tl_sabra_model.py --time_to_run=1
"""
import sys

sys.path.append("..")
import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.saving.save_data_funcs as g_save
import numpy as np
import shell_model_experiments.sabra_model.runge_kutta4 as rk4
from numba import njit, types
from pyinstrument import Profiler
from shell_model_experiments.params.params import *

profiler = Profiler()

# Set global params
cfg.GLOBAL_PARAMS.record_max_time = 30
cfg.GLOBAL_PARAMS.ref_run = False


@njit(
    (types.Array(types.complex128, 1, "C", readonly=False))(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 2, "C", readonly=False),
        types.int64,
        types.float64,
        types.float64,
        types.float64,
        types.Array(types.complex128, 2, "C", readonly=True),
    ),
    cache=cfg.NUMBA_CACHE,
)
def run_model(
    u_tl_old: np.ndarray,
    u_ref: np.ndarray,
    du_array: np.ndarray,
    data_out: np.ndarray,
    Nt_local: int,
    ny: float,
    diff_exponent: int,
    forcing: float,
    pre_factor: np.ndarray,
):
    """Execute the integration of the sabra shell model.

    Parameters
    ----------
    u_tl_old : ndarray
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
            data_out[sample_number, 1:] = (
                u_ref[bd_size:-bd_size] + u_tl_old[bd_size:-bd_size]
            )
            sample_number += 1

        # Solve the TL model
        u_tl_old: np.ndarray = rk4.tl_runge_kutta4(
            y0=u_tl_old,
            h=dt,
            du=du_array,
            u_ref=u_ref,
            diff_exponent=diff_exponent,
            local_ny=ny,
            prefactor=pre_factor,
        )

        # Solve nonlinear equation to get reference velocity
        u_ref = rk4.runge_kutta4(
            y0=u_ref,
            h=dt,
            du=du_array,
            forcing=forcing,
        )

        # Solve linear diffusive term explicitly
        u_ref[bd_size:-bd_size] = u_ref[bd_size:-bd_size] * np.exp(
            -ny * k_vec_temp ** diff_exponent * dt
        )

    return u_tl_old


def main(args=None):

    # Import reference data
    u_ref, _, ref_header_dict = g_import.import_start_u_profiles(args=args)

    if u_ref.size > sdim + 2 * bd_size:
        raise ValueError(
            "To many reference u profiles imported for the TL. Only one is needed"
        )
    else:
        u_ref = u_ref.ravel()

    profiler.start()
    print(f'\nRunning TL sabra model for {args["Nt"]*dt:.2f}s')

    # Prepare data out array and prefactor
    data_out = np.zeros(
        (int(args["Nt"] * sample_rate), n_k_vec + 1), dtype=np.complex128
    )
    prefactor_reshaped = np.reshape(pre_factor, (-1, 1))

    # Run model
    run_model(
        u_ref,
        du_array,
        data_out,
        args["Nt"],
        args["ny"],
        args["diff_exponent"],
        args["forcing"],
        prefactor_reshaped,
    )

    if not args["skip_save_data"]:
        print(f"saving record\n")
        save_path = g_save.save_data(data_out, args=args, prefix="tl_")

    profiler.stop()
    print(profiler.output_text())


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    ref_arg_setup = a_parsers.RelReferenceArgSetup()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    args["ny"] = ny_from_ny_n_and_forcing(
        args["forcing"], args["ny_n"], args["diff_exponent"]
    )

    args["Nt"] = int(args["time_to_run"] / dt)

    main(args=args)
