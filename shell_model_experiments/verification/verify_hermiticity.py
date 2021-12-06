"""Script to verify if the J*J matrix is hermitian (where J is the jacobian of 
the shell model)
"""
import sys

import matplotlib.pyplot as plt
import seaborn as sb
from pyinstrument import Profiler

sys.path.append("..")
import copy
import pathlib as pl

import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.user_interface as g_ui
import numpy as np
from general.params.model_licences import Models

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
    import shell_model_experiments.utils.special_params as sh_sparams
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
    sparams = sh_sparams
# elif cfg.MODEL == Models.LORENTZ63:
#     import lorentz63_experiments.params.params as l63_params
#     import lorentz63_experiments.params.special_params as l63_sparams

#     params = l63_params
#     sparams = l63_sparams


def verify_hermiticity(args):

    # Import reference data
    (
        u_init_profiles,
        _,
        _,
    ) = g_import.import_start_u_profiles(args=args)

    (
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    ) = sh_nm_estimator.init_jacobian()

    n_runs = int(args["n_profiles"] * args["n_runs_per_profile"])
    JJ_matrix_store = np.empty((n_runs, params.sdim, params.sdim), dtype=np.complex128)

    for i in range(n_runs):
        sh_nm_estimator.calc_jacobian(
            np.copy(u_init_profiles[:, i]),
            args["diff_exponent"],
            args["ny"],
            params,
            diagonal0,
            diagonal1,
            diagonal2,
            diagonal_1,
            diagonal_2,
        )

        copy_J_matrix = np.copy(J_matrix)

        adjoint_J_matrix = sh_nm_estimator.calc_adjoint_jacobian(
            np.copy(u_init_profiles[:, i]),
            args["diff_exponent"],
            args["ny"],
            params,
            J_matrix,
            diagonal0,
            diagonal1,
            diagonal2,
            diagonal_1,
            diagonal_2,
        )

        JJ_matrix_store[i, :, :] = adjoint_J_matrix @ copy_J_matrix

    JJ_matrix_adjoint_store = np.transpose(JJ_matrix_store, axes=(0, 2, 1)).conj()

    mean_difference = np.abs(np.mean(JJ_matrix_store - JJ_matrix_adjoint_store, axis=0))

    sb.heatmap(
        mean_difference,
        cmap="Purples",
        cbar_kws={"label": "$\\langle J^*J - JJ^* \\rangle$"},
    )
    plt.title("Verify hermiticity of J*J matrix")
    plt.xlabel("Shell number")
    plt.ylabel("Shell number")
    plt.show()


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

        # Add ny argument
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )

    args["Nt"] = int(args["time_to_run"] / params.dt)

    verify_hermiticity(args)

    # g_plt_utils.save_or_show_plot(args)
