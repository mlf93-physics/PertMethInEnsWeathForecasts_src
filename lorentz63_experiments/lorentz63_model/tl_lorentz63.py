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
import general.utils.argument_parsers as a_parsers
import config as cfg

import matplotlib.pyplot as plt

profiler = Profiler()

# Set global params
cfg.GLOBAL_PARAMS.record_max_time = 3000


@njit(
    (
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=True),
        types.Array(types.float64, 2, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.int64,
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def run_model(u_old, du_array, u_ref, deriv_matrix, data_out, Nt_local, r_const):
    """Execute the integration of the tangent linear Lorentz63 model.

    Parameters
    ----------
    u_old : ndarray
        The initial lorentz velocities
    du_array : ndarray
        A helper array used to store the current derivative of the lorentz
        velocities.
    data_out : ndarray
        An array to store samples of the integrated lorentz_velocities.

    """

    sample_number = 0
    # Perform calculations
    for i in range(Nt_local):
        # Save samples for plotting
        if i % int(1 / sample_rate) == 0:
            data_out[sample_number, 0] = dt * i
            data_out[sample_number, 1:] = u_old
            sample_number += 1

        # Update u_old
        u_old = rk4.tl_runge_kutta4(
            y0=u_old,
            h=dt,
            du=du_array,
            u_ref=u_ref[i, :],
            deriv_matrix=deriv_matrix,
            r_const=r_const,
        )


def main(args=None):

    # Import reference data
    _, u_ref, ref_header_dict = g_import.import_ref_data(args=args)

    deriv_matrix = l63_nm_estimator.init_jacobian(args)

    profiler.start()
    print(f'\nRunning Lorentz63 TL model for {args["Nt"]*dt:.2f}s\n')

    data_out = np.zeros((args["Nt"], sdim + 1), dtype=np.float64)

    # Run model
    run_model(
        u_ref[0, :],
        du_array,
        u_ref,
        deriv_matrix,
        data_out,
        args["Nt"],
        r_const=args["r_const"],
    )

    if not args["skip_save_data"]:
        print(f"saving record\n")
        g_save.save_data(data_out, args=args)

    profiler.stop()
    print(profiler.output_text(color=True))


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_arg_setup = a_parsers.StandardRunnerArgSetup()
    stand_arg_setup.setup_parser()
    ref_arg_setup = a_parsers.ReferenceAnalysisArgParser()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    # Add/edit arguments
    args["Nt"] = int(args["time_to_run"] / dt)

    main(args)
