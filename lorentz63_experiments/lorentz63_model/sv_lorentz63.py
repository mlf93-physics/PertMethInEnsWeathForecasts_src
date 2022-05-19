import sys

sys.path.append("..")
import math

import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.perturb_utils as pt_utils
import general.utils.saving.save_data_funcs as g_save
import general.utils.util_funcs as g_utils
import lorentz63_experiments.lorentz63_model.runge_kutta4 as rk4
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
import lorentz63_experiments.utils.util_funcs as ut_funcs
from tl_lorentz63 import run_model as tl_model
from atl_lorentz63 import run_model as atl_model
import matplotlib.pyplot as plt
import numpy as np
from lorentz63_experiments.params.params import *
from numba import njit, types
from pyinstrument import Profiler

profiler = Profiler()

# Set global params

cfg.GLOBAL_PARAMS.ref_run = False


def run_model(
    u_tl_old,
    u_ref_old,
    lorentz_matrix,
    jacobian_matrix,
    data_out_tl,
    data_out_atl,
    Nt_local,
    r_const,
    n_iterations,
):
    print("u_ref_old", u_ref_old)

    for i in range(n_iterations):
        print("i", i)
        tl_model(
            u_tl_old,
            u_ref_old,
            lorentz_matrix,
            jacobian_matrix,
            data_out_tl,
            Nt_local,
            r_const,
            raw_perturbation=True,
        )

        print("data_out[-1, 1:] tlm", data_out_tl[-1, 1:])
        input()

        atl_model(
            np.reshape(data_out_tl[-1, 1:], (sdim, 1)),
            u_ref_old,
            lorentz_matrix,
            jacobian_matrix,
            data_out_atl,
            Nt_local,
            r_const,
            raw_perturbation=True,
        )

        print("data_out_atl[0, 1:] atlm", data_out_atl[0, 1:])
        input()

        # Rescale last data point to norm 1 to get new u_tl_old
        u_tl_old = data_out_atl[0, 1:] / np.linalg.norm(data_out_atl[0, 1:], axis=0)

    return data_out_atl[0, 1:]


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
    u_tl_old = g_utils.normalize_array(perturb, norm_value=1)
    print("u_tl_old start", u_tl_old)

    # Initialise jacobian and deriv matrix
    jacobian_matrix = l63_nm_estimator.init_jacobian(args)
    lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)

    profiler.start()
    print(f'\nRunning Lorentz63 TL model for {args["Nt"]*dt:.2f}s\n')

    data_out_tl = np.zeros((args["Nt"], sdim + 1), dtype=np.float64)
    data_out_atl = np.zeros((args["Nt"], sdim + 1), dtype=np.float64)

    n_iterations = 30

    # Run model
    out_array = run_model(
        u_tl_old,
        u_ref,
        lorentz_matrix,
        jacobian_matrix,
        data_out_tl,
        data_out_atl,
        args["Nt"],
        args["r_const"],
        n_iterations,
    )
    print("out_array", out_array)

    # if not args["skip_save_data"]:
    #     print(f"saving record\n")
    #     g_save.save_data(data_out_atl, args=args, prefix="tl_")

    profiler.stop()
    print(profiler.output_text(color=True))


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    ref_arg_setup = a_parsers.RelReferenceArgSetup()
    ref_arg_setup.setup_parser()
    ref_arg_setup.validate_arguments()
    args = ref_arg_setup.args

    # Add/edit arguments
    args["Nt"] = int(args["time_to_run"] / dt) + 1

    main(args)
