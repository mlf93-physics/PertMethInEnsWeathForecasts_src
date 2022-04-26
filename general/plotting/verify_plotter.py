import sys

sys.path.append("..")
import pathlib as pl

import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import general.plotting.plot_config as plt_config
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp_optim
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from matplotlib.ticker import MultipleLocator
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


def logfunction(x, a, b):
    return a * np.log(x) + b


def linearfunction(x, a, b):
    return a * x + b


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
    mean_error_norm = np.mean(np.log(error_norms), axis=1)
    std_error_norm = np.std(np.log(error_norms), axis=1)

    # Get time
    time_array = verification_data[:, 0].real

    if cfg.MODEL == cfg.Models.LORENTZ63:
        logfit_split_index = 50
        linfit_split_index = 100
    elif cfg.MODEL == cfg.Models.SHELL_MODEL:
        if args["regime_start"] == "low":
            logfit_split_index = 6
            linfit_split_index = 20
        elif args["regime_start"] == "high":
            logfit_split_index = 40
            linfit_split_index = 80

    # Fit data
    log_popt, log_pcov = sp_optim.curve_fit(
        logfunction,
        np.repeat(time_array[1:logfit_split_index], error_norms.shape[1]).ravel(),
        np.log(error_norms[1:logfit_split_index, :]).ravel(),
    )
    print("log_popt", log_popt, "log_pcov", log_pcov)

    lin_popt, lin_pcov = sp_optim.curve_fit(
        linearfunction,
        np.repeat(time_array[linfit_split_index:], error_norms.shape[1]).ravel(),
        np.log(error_norms[linfit_split_index:, :]).ravel(),
    )
    print("lin_popt", lin_popt, "lin_pcov", lin_pcov)

    axes.plot(
        time_array[1:],
        mean_error_norm[1:],
        "k-",
        label="$\\langle ||\\delta x(t)|| \\rangle$",
    )
    axes.plot(
        time_array[1:],
        logfunction(time_array[1:], *log_popt),
        label="Log. fit ($f(x)=a \\mathrm{{log}}(x) + b$)",
        color="b",
        linestyle="dashed",
    )
    axes.plot(
        time_array,
        linearfunction(time_array, *lin_popt),
        label="Linear fit",
        color="b",
        linestyle="dotted",
    )
    axes.set_ylabel(
        "Log error norm $\\mathrm{{log}}||E(t)||$"  # $\\mathrm{{log}}||L \\delta x(0)$ - $( M[x_0 + \\delta x(0)] - M(x_0))||$"
    )
    axes.set_xlabel("Time")
    axes.set_title("TLM error verification")
    axes.legend()

    fig.subplots_adjust(
        top=0.98, bottom=0.199, left=0.25, right=0.944, hspace=0.2, wspace=0.2
    )

    if args["tolatex"]:
        axes.get_legend().remove()
        plt_config.adjust_axes(axes)

    if args["save_fig"]:
        if cfg.MODEL == Models.SHELL_MODEL:
            subfolder = "shell"
            file_name = "verification_tlm_" + args["regime_start"]
        elif cfg.MODEL == Models.LORENTZ63:
            subfolder = "l63"
            file_name = "verification_tlm"

        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/numerical_setup/" + subfolder,
            file_name=file_name,
        )


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

    # Make profiler
    # profiler = Profiler()
    # profiler.start()
    plt_config.adjust_default_fig_axes_settings(args)

    if "tl_error_verification" in args["plot_type"]:
        plot_tl_error_verification(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    # profiler.stop()
    # print(profiler.output_text(color=True))
    g_plt_utils.save_or_show_plot(args)
