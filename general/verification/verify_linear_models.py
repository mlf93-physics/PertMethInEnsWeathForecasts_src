import sys

sys.path.append("..")
import math
import matplotlib.pyplot as plt
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
from general.params.model_licences import Models
import config as cfg

import matplotlib.pyplot as plt

if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR
    from shell_model_experiments.params.params import ParamsStructType
    from shell_model_experiments.sabra_model.sabra_model import run_model as sh_model
    from shell_model_experiments.sabra_model.tl_sabra_model import (
        run_model as sh_tl_model,
    )

    # Get parameters for model
    params = PAR
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams
    from lorentz63_experiments.lorentz63_model.lorentz63 import run_model as l63_model
    from lorentz63_experiments.lorentz63_model.tl_lorentz63 import (
        run_model as l63_tl_model,
    )

    # Get parameters for model
    params = l63_params
    sparams = l63_sparams

profiler = Profiler()


def run_l63_tl_model_verification(
    args,
    u_ref,
    u_perturb,
    jacobian_matrix,
    lorentz_matrix,
    tl_data_out,
    nl_model_pert_data_out,
    nl_model_non_pert_data_out,
):
    print(f"\nRunning verification of the Lorentz63 TL model\n")

    # Run TL model
    l63_tl_model(
        u_perturb,
        du_array,
        u_ref,
        lorentz_matrix,
        jacobian_matrix,
        tl_data_out,
        args["Nt"],
        r_const=args["r_const"],
        raw_perturbation=True,
    )

    # Run non-linear model on perturbed start velocity
    l63_model(
        u_ref + u_perturb,
        du_array,
        lorentz_matrix,
        nl_model_pert_data_out,
        args["Nt"],
    )

    # Run non-linear model on non-perturbed start velocity
    l63_model(
        u_ref,
        du_array,
        lorentz_matrix,
        nl_model_non_pert_data_out,
        args["Nt"],
    )


def verify_tlm_model(args):
    # Import reference data
    u_ref, _, ref_header_dict = g_import.import_start_u_profiles(args=args)

    if u_ref.size > sdim:
        raise ValueError(
            "To many reference u profiles imported for the verification of the TL. Only one is needed"
        )
    else:
        u_ref = u_ref.ravel()

    # Prepare random perturbation
    perturb = pt_utils.generate_rd_perturbations()
    # Normalize perturbation
    u_perturb = g_utils.normalize_array(perturb, norm_value=seeked_error_norm)

    if cfg.MODEL == Models.SHELL_MODEL:
        pass
    elif cfg.MODEL == Models.LORENTZ63:
        # Initialise jacobian and deriv matrix
        jacobian_matrix = l63_nm_estimator.init_jacobian(args)
        lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)
        tl_data_out = np.zeros((args["Nt"], sdim + 1), dtype=np.float64)
        nl_model_pert_data_out = np.zeros((args["Nt"], sdim + 1), dtype=np.float64)
        nl_model_non_pert_data_out = np.zeros((args["Nt"], sdim + 1), dtype=np.float64)

        run_l63_tl_model_verification(
            args,
            u_ref,
            u_perturb,
            jacobian_matrix,
            lorentz_matrix,
            tl_data_out,
            nl_model_pert_data_out,
            nl_model_non_pert_data_out,
        )

        tl_error_norm = np.linalg.norm(tl_data_out[:, 1:], axis=1)
        nl_error_norm = np.linalg.norm(
            nl_model_pert_data_out[:, 1:] - nl_model_non_pert_data_out[:, 1:], axis=1
        )

        plt.plot(tl_data_out[:, 0], tl_error_norm, label="tl_error_norm")
        plt.plot(tl_data_out[:, 0], nl_error_norm, label="nl_error_norm")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    ref_arg_setup = a_parsers.RelReferenceArgSetup()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    # Add/edit arguments
    args["Nt"] = int(args["time_to_run"] / dt)

    profiler.start()

    verify_tlm_model(args)

    profiler.stop()
    print(profiler.output_text(color=True))
