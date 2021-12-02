import sys

sys.path.append("..")
import pathlib as pl

import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp_optim
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from pyinstrument import Profiler

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params

    params = l63_params


def parabola_order2(x, a, b):
    return a * x ** 2 + b * x


def plot_tl_error_verification(args, axes=None):

    # Find datafiles
    (
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        perturb_file_names,
    ) = g_import.imported_sorted_perturbation_info(
        args["exp_folder"],
        args,
        search_pattern="verification*.csv",
    )

    # Prepare axes
    if axes is None:
        fig, axes = plt.subplots(1, 1)

    # Import error data and calculate norms
    error_norms = []
    for file_name in perturb_file_names:
        verification_data, header_dict = g_import.import_data(file_name)

        error_norms.append(np.linalg.norm(verification_data[:, 1:], axis=1))

    error_norms = np.array(error_norms).T

    # Get mean and std
    mean_error_norm = np.mean(error_norms, axis=1)
    std_error_norm = np.std(error_norms, axis=1)

    # Get time
    time_array = verification_data[:, 0].real

    # Fit data
    popt, pcov = sp_optim.curve_fit(
        parabola_order2,
        np.repeat(time_array, error_norms.shape[1]).ravel(),
        error_norms.ravel(),
    )

    axes.plot(
        time_array, mean_error_norm, "b-", label="$\\langle ||\\delta x(t)|| \\rangle$"
    )
    axes.plot(
        time_array,
        mean_error_norm + std_error_norm,
        "b--",
        label="$\\langle ||\\delta x(t)|| \\rangle + \\sigma$",
    )
    axes.plot(
        time_array,
        parabola_order2(time_array, *popt),
        label="Poly. fit ($f(x)=ax^2 + bx$)",
    )
    axes.set_ylabel(
        "Error difference\n $\\mathbf{{L}}\\delta x(0)$ - $( M[x_0 + \\delta x(0)] - M(x_0))$"
    )
    axes.set_xlabel("Time")
    axes.set_title("TLM error verification")
    axes.legend()


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
        sh_utils.update_arrays(params)

    # Make profiler
    profiler = Profiler()
    profiler.start()

    if "tl_error_verification" in args["plot_type"]:
        plot_tl_error_verification(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    profiler.stop()
    print(profiler.output_text(color=True))
    g_plt_utils.save_or_show_plot(args)
