"""Solve the ATL sabra shell model

Example
-------
python sabra_model/atl_sabra_model.py --time_to_run=1
"""
import sys

sys.path.append("..")
import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.saving.save_data_funcs as g_save
import general.utils.util_funcs as g_utils
import general.utils.perturb_utils as pt_utils
import numpy as np
import shell_model_experiments.sabra_model.runge_kutta4 as rk4
import shell_model_experiments.utils.special_params as sparams
import shell_model_experiments.utils.util_funcs as ut_funcs
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
from numba import njit, types, typeof
from pyinstrument import Profiler
from shell_model_experiments.params.params import PAR
from shell_model_experiments.params.params import ParamsStructType

profiler = Profiler()

# Set global params
cfg.GLOBAL_PARAMS.record_max_time = 30
cfg.GLOBAL_PARAMS.ref_run = False


@njit(
    [
        (types.Array(types.complex128, 1, "C", readonly=False))(
            types.Array(types.complex128, 1, "C", readonly=False),
            types.Array(types.complex128, 1, "C", readonly=False),
            types.Array(types.complex128, 2, "C", readonly=False),
            types.int64,
            types.float64,
            types.float64,
            types.float64,
            typeof(PAR),
            types.Array(types.complex128, 2, "C", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.boolean,
        ),
        (types.Array(types.complex128, 1, "C", readonly=False))(
            types.Array(types.complex128, 1, "C", readonly=False),
            types.Array(types.complex128, 1, "C", readonly=False),
            types.Array(types.complex128, 2, "C", readonly=False),
            types.int64,
            types.float64,
            types.float64,
            types.float64,
            typeof(PAR),
            types.Array(types.complex128, 2, "C", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Array(types.complex128, 1, "A", readonly=False),
            types.Omitted(False),
        ),
    ],
    cache=cfg.NUMBA_CACHE,
)
def run_model(
    u_atl_old: np.ndarray,
    u_ref_old: np.ndarray,
    data_out: np.ndarray,
    Nt_local: int,
    ny: float,
    diff_exponent: int,
    forcing: float,
    PAR: ParamsStructType,
    J_matrix: np.ndarray,
    diagonal0: np.ndarray,
    diagonal1: np.ndarray,
    diagonal2: np.ndarray,
    diagonal_1: np.ndarray,
    diagonal_2: np.ndarray,
    raw_perturbation: bool = False,
):
    """Execute the integration of the adjoint of the TL sabra shell model.

    Parameters
    ----------
    u_atl_old : ndarray
        The initial shell velocity profile
    du_array : ndarray
        A helper array used to store the current derivative of the shell
        velocities.
    data_out : ndarray
        An array to store samples of the integrated shell velocities.

    """
    # Prepare prefactor and ref_data array
    ref_data = np.zeros((Nt_local, PAR.sdim + 2 * PAR.bd_size), dtype=sparams.dtype)

    # Run forward non-linear model
    for i in range(Nt_local):
        # Save reference data
        ref_data[i, PAR.bd_size : -PAR.bd_size] = u_ref_old[PAR.bd_size : -PAR.bd_size]
        # Solve nonlinear equation to get reference velocity
        u_ref_old = rk4.runge_kutta4(u_old=u_ref_old, forcing=forcing, PAR=PAR)

        # Solve linear diffusive term explicitly
        u_ref_old[PAR.bd_size : -PAR.bd_size] = u_ref_old[
            PAR.bd_size : -PAR.bd_size
        ] * np.exp(-ny * PAR.k_vec_temp ** diff_exponent * PAR.dt)

    sample_number = -1
    # Run backward ATL model
    for i in range(Nt_local - 1, -1, -1):
        if i % int(1 / PAR.sample_rate) == 0:
            # Save samples of calculation
            data_out[sample_number, 0] = PAR.dt * i + 0j
            data_out[sample_number, 1:] = u_atl_old[PAR.bd_size : -PAR.bd_size]
            # Add reference data if requested
            if not raw_perturbation:
                # Add reference data to ATL model trajectory, since only the perturbation
                # is integrated in the model
                data_out[sample_number, 1:] += ref_data[i, :]

            sample_number -= 1

        # Break if last datapoint has been saved
        if i == 0:
            break

        # Solve the ATL model
        u_atl_old: np.ndarray = rk4.atl_runge_kutta4(
            u_atl_old,
            ref_data[i - 1, :],
            diff_exponent,
            ny,
            forcing,
            PAR,
            J_matrix,
            diagonal0,
            diagonal1,
            diagonal2,
            diagonal_1,
            diagonal_2,
        )

    return u_atl_old


def main(args=None):

    # Import reference data
    u_ref, _, ref_header_dict = g_import.import_start_u_profiles(args=args)

    if u_ref.size > PAR.sdim + 2 * PAR.bd_size:
        raise ValueError(
            "To many reference u profiles imported for the TL. Only one is needed"
        )
    else:
        u_ref = u_ref.ravel()

    # Prepare random perturbation
    perturb = pt_utils.generate_rd_perturbations()
    # Normalize perturbation
    u_atl_old = g_utils.normalize_array(perturb, norm_value=PAR.seeked_error_norm)
    # Pad array
    u_atl_old = np.pad(
        u_atl_old,
        pad_width=((PAR.bd_size, PAR.bd_size)),
        mode="constant",
    )
    # u_atl_old = np.expand_dims(u_atl_old, axis=1)

    # Initialise the Jacobian and diagonal arrays
    (
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    ) = sh_nm_estimator.init_jacobian()

    profiler.start()
    print(f'\nRunning ATL sabra model for {args["Nt"]*PAR.dt:.2f}s')

    # Prepare data out array and prefactor
    data_out = np.zeros(
        (int(args["Nt"] * PAR.sample_rate), PAR.sdim + 1), dtype=sparams.dtype
    )

    # Run model
    run_model(
        u_atl_old,
        u_ref,
        data_out,
        args["Nt"],
        args["ny"],
        args["diff_exponent"],
        args["forcing"],
        PAR,
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    )

    if not args["skip_save_data"]:
        print(f"saving record\n")
        save_path = g_save.save_data(data_out, args=args, prefix="atl_")

    profiler.stop()
    print(profiler.output_text(color=True))


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    ref_arg_setup = a_parsers.RelReferenceArgSetup()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    # Initiate and update variables and arrays
    ut_funcs.update_dependent_params(PAR)
    ut_funcs.set_params(PAR, parameter="sdim", value=args["sdim"])
    ut_funcs.update_arrays(PAR)
    args["ny"] = ut_funcs.ny_from_ny_n_and_forcing(
        args["forcing"], args["ny_n"], args["diff_exponent"]
    )

    args["Nt"] = int(args["time_to_run"] / PAR.dt)

    main(args=args)
