import sys

sys.path.append("..")
import math
from pyinstrument import Profiler
import numpy as np
from numba import njit, types
import lorentz63_experiments.lorentz63_model.runge_kutta4 as rk4
from lorentz63_experiments.params.params import *
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
import general.utils.saving.save_data_funcs as g_save
import general.utils.importing.import_data_funcs as g_import
import lorentz63_experiments.utils.util_funcs as ut_funcs
import general.utils.util_funcs as g_utils
import general.utils.perturb_utils as pt_utils
import general.utils.argument_parsers as a_parsers
import config as cfg

import matplotlib.pyplot as plt

profiler = Profiler()

# Set global params
cfg.GLOBAL_PARAMS.record_max_time = 3000
cfg.GLOBAL_PARAMS.ref_run = False


# @njit(
#     (
#         types.Array(types.float64, 1, "C", readonly=False),
#         types.Array(types.float64, 1, "C", readonly=False),
#         types.Array(types.float64, 1, "C", readonly=False),
#         types.Array(types.float64, 2, "C", readonly=False),
#         types.Array(types.float64, 2, "C", readonly=False),
#         types.Array(types.float64, 2, "C", readonly=False),
#         types.int64,
#         types.float64,
#     ),
#     cache=cfg.NUMBA_CACHE,
# )
def run_model(
    u_atl_old,
    du_array,
    u_ref_old,
    lorentz_matrix,
    jacobian_matrix,
    data_out,
    ref_data,
    Nt_local,
    r_const,
):
    """Execute the integration of the tangent linear Lorentz63 model.

    Parameters
    ----------
    u_atl_old : ndarray
        The initial lorentz perturbation velocities
    du_array : ndarray
        A helper array used to store the current derivative of the lorentz
        velocities.
    data_out : ndarray
        An array to store samples of the integrated lorentz_velocities.

    """

    # Run forward non-linear model
    for i in range(Nt_local):
        # Update u_ref_old
        u_ref_old = rk4.runge_kutta4(
            y0=u_ref_old, h=dt, du_array=du_array, lorentz_matrix=lorentz_matrix
        )
        ref_data[i, :] = u_ref_old

    # Run backward ATL model
    for i in range(Nt_local - 1, -1, -1):
        # Save samples for plotting
        data_out[i, 0] = dt * i
        # Add reference data to TL model trajectory, since only the perturbation
        # is integrated in the model
        data_out[i, 1:] = u_atl_old.ravel()

        # Update u_atl_old
        u_atl_old = rk4.atl_runge_kutta4(
            u_atl_old=u_atl_old,
            dt=dt,
            du_array=du_array,
            u_ref_old=ref_data[i - 1, :],
            lorentz_matrix=lorentz_matrix,
            jacobian_matrix=jacobian_matrix,
            r_const=r_const,
        )


def main(args=None):

    # Import reference data
    u_ref, _, ref_header_dict = g_import.import_start_u_profiles(args=args)

    if u_ref.size > sdim:
        raise ValueError(
            "To many reference u profiles imported for the TL. Only one is needed"
        )
    else:
        u_ref = u_ref.ravel()

    # Prepare random perturbation
    perturb = pt_utils.generate_rd_perturbations()
    # Normalize perturbation
    u_atl_old = g_utils.normalize_array(perturb, norm_value=seeked_error_norm)
    u_atl_old = np.reshape(u_atl_old, (sdim, 1))

    # Initialise jacobian and deriv matrix
    jacobian_matrix = l63_nm_estimator.init_jacobian(args)
    lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)

    profiler.start()
    print(f'\nRunning Lorentz63 ATL model for {args["Nt"]*dt:.2f}s\n')

    data_out = np.zeros((args["Nt"], sdim + 1), dtype=np.float64)
    ref_data = np.zeros((args["Nt"], sdim), dtype=np.float64)

    # Run model
    run_model(
        u_atl_old,
        du_array,
        u_ref,
        lorentz_matrix,
        jacobian_matrix,
        data_out,
        ref_data,
        args["Nt"],
        r_const=args["r_const"],
    )

    if not args["skip_save_data"]:
        print(f"saving record\n")
        g_save.save_data(data_out, args=args, prefix="atl_")

    profiler.stop()
    print(profiler.output_text(color=True))


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    ref_arg_setup = a_parsers.RelReferenceArgSetup()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    # Add/edit arguments
    args["Nt"] = int(args["time_to_run"] / dt)

    main(args)
