"""Make plots to compare the different perturbation methods

Example
-------
python ../general/plotting/plot_comparisons.py
--plot_type=error_norm_compare
--exp_folder=test1_new_params

"""

import sys

sys.path.append("..")
import math
import re
import pathlib as pl
import copy
import config as cfg
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.arg_utils as a_utils
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.importing.import_utils as g_imp_utils
import general.utils.plot_utils as g_plt_utils
import general.analyses.plot_analyses as g_plt_anal
from general.plotting.plot_params import *
import general.plotting.plot_config as plt_config
from general.utils.module_import.type_import import *
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import general.utils.experiments.exp_utils as e_utils
import general.utils.running.runner_utils as r_utils
from libs.libutils import type_utils as lib_type_utils
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import seaborn as sb
from general.params.model_licences import Models
import general.plotting.plot_comparisons as plt_compare

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.plotting.plot_data as sh_plot
    import shell_model_experiments.utils.util_funcs as sh_utils
    import shell_model_experiments.utils.special_params as sh_sparams
    import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.plotting.plot_data as l63_plot

    params = l63_params
    sparams = l63_sparams


def collect_error_norm_plots(args):
    # Make axes
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    # Plot BV, BV-EOF, SV, LV
    args["perturbations"] = ["bv", "bv_eof", "sv", "lv"]
    copy_exp_folder = copy.deepcopy(args["exp_folder"])
    twin_axes = plt_compare.plot_error_norm_comparison(args, axes=axes[0])
    plt_config.hide_axis_labels(twin_axes)

    # Plot RD, NM, RF
    args["perturbations"] = ["rd", "nm", "rf"]
    # Reset exp_folder
    args["exp_folder"] = copy_exp_folder
    twin_axes = plt_compare.plot_error_norm_comparison(args, axes=axes[1])
    plt_config.hide_axis_labels(twin_axes)

    g_plt_utils.add_subfig_labels(axes)

    if cfg.MODEL == cfg.Models.LORENTZ63:
        ylabel_suffix = "$||\\mathbf{x} - \\mathbf{x}_{ref}||$"
        ylabel_right_suffix = "$\\frac{1}{2}\\sum_i {x}_{i, ref}^2$"
    elif cfg.MODEL == cfg.Models.SHELL_MODEL:
        ylabel_suffix = "$||\\mathbf{u} - \\mathbf{u}_{ref}||$"
        ylabel_right_suffix = "$\\frac{1}{2} u_{n, ref} u_{n, ref}^*$"

    label_axes: plt.Axes = fig.add_subplot(111, frame_on=False)
    label_axes.tick_params(
        labelcolor="none", bottom=False, left=False, right=False, top=False
    )
    label_axes.set_xlabel("Time", labelpad=25)
    label_axes.set_ylabel("Error, " + ylabel_suffix, labelpad=30)

    label_axes_right = label_axes.twinx()
    label_axes_right.tick_params(
        labelcolor="none", bottom=False, left=False, right=False, top=False
    )
    label_axes_right.set_ylabel("Energy, " + ylabel_right_suffix, labelpad=30)

    plt_config.hide_axis_ticks([label_axes, label_axes_right])

    # Remove spines
    for spine, spine_right_axes in zip(label_axes.spines, label_axes_right.spines):
        label_axes.spines[spine].set_visible(False)
        label_axes_right.spines[spine_right_axes].set_visible(False)

    fig.subplots_adjust(left=0.120, bottom=0.120, right=0.880, top=0.940)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            subpath="thesis_figures/results_and_analyses/l63/",
            file_name="compare_error_norm",
        )


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args: dict = compare_plot_arg_parser.args

    a_utils.react_on_comparison_arguments(args)

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

    g_ui.confirm_run_setup(args)
    plt_config.adjust_default_fig_axes_settings(args)

    if "collect_error_norm_compare" in args["plot_type"]:
        collect_error_norm_plots(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args, tight_layout_rect=[0, 0, 0.9, 1])
