"""Make plots to compare the different perturbation methods

Example
-------
python ../general/plotting/plot_comparisons.py
--plot_type=error_norm_compare
--exp_folder=test1_new_params

"""

import sys

sys.path.append("..")
import copy
import pathlib as pl
import numpy as np
import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.arg_utils as a_utils
import general.utils.plot_utils as g_plt_utils
from general.plotting.plot_params import *
import general.plotting.plot_config as plt_config
from general.utils.module_import.type_import import *
import general.utils.user_interface as g_ui
import matplotlib.pyplot as plt
from general.params.model_licences import Models
import general.plotting.plot_comparisons as plt_compare

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.util_funcs as sh_utils
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR as PAR_SH

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    params = l63_params
    sparams = l63_sparams


def collect_exp_growth_rate_plots(args):
    if cfg.MODEL == cfg.Models.SHELL_MODEL:
        specific_runs_per_profile_dict = {
            "bv": None,
            "rd": None,
            "nm": None,
            "rf": None,
            "bv_eof": [0, 9, 17],
            "sv": [0, 9, 17],
            "lv": [1, 9, 17],
        }
    else:
        specific_runs_per_profile_dict = None
    # Make axes
    fig, axes = plt.subplots(nrows=2, ncols=1)

    copy_args1 = copy.deepcopy(args)
    copy_args1["exp_folder"] = "low_pred/compare_pert_exp_growth_rate_it0.0005"

    plt_compare.plot_exp_growth_rate_comparison(
        copy_args1,
        axes=axes[0],
        specific_runs_per_profile_dict=specific_runs_per_profile_dict,
    )
    axes[0].set_xlim(-0.0001, 0.005)
    axes[0].set_ylim(-2000, None)
    axes[0].set_yticks([-1000, 0, 1000])
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    copy_args2 = copy.deepcopy(args)
    copy_args2["exp_folder"] = "high_pred/compare_pert_exp_growth_rate_it0.004"

    plt_compare.plot_exp_growth_rate_comparison(
        copy_args2,
        axes=axes[1],
        specific_runs_per_profile_dict=specific_runs_per_profile_dict,
    )
    axes[1].set_xlim(-0.0001, 0.01)
    axes[1].set_ylim(-1000, 250)
    axes[1].set_yticks(np.array([-1.0, -0.5, 0.0], dtype=np.float32) * 1e3)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.subplots_adjust(
        top=0.949, bottom=0.126, left=0.111, right=0.966, hspace=0.395, wspace=0.2
    )

    if args["tolatex"]:
        plt_config.remove_legends(axes)
        plt_config.adjust_axes(axes)
        g_plt_utils.add_subfig_labels(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath=pl.Path(
                "thesis_figures",
                "results_and_analyses/shell/"
                if args["save_sub_folder"] is None
                else args["save_sub_folder"],
            ),
            file_name="compare_instant_exp_growth_rates"
            if args["save_fig_name"] is None
            else args["save_fig_name"],
        )


def collect_sv_exp_growth_rate_plots(args):

    specific_runs_per_profile_dict = None

    # Make axes
    fig, axes = plt.subplots(nrows=2, ncols=1)

    copy_args1 = copy.deepcopy(args)
    copy_args1["exp_folder"] = "low_pred/compare_pert_exp_growth_rate_it0.0005"

    plt_compare.plot_exp_growth_rate_comparison(
        copy_args1,
        axes=axes[0],
        specific_runs_per_profile_dict=specific_runs_per_profile_dict,
        highlight_perts=[0, 1],
    )
    axes[0].set_xlim(-0.0001, 0.005)
    axes[0].set_ylim(-700, 700)
    axes[0].set_yticks([-500, 0, 500])
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    copy_args2 = copy.deepcopy(args)
    copy_args2["exp_folder"] = "high_pred/compare_pert_exp_growth_rate_it0.004"

    plt_compare.plot_exp_growth_rate_comparison(
        copy_args2,
        axes=axes[1],
        specific_runs_per_profile_dict=specific_runs_per_profile_dict,
        highlight_perts=[0, 3],
    )
    axes[1].set_xlim(-0.0001, 0.01)
    axes[1].set_ylim(-500, 150)
    axes[1].set_yticks(np.array([-0.5, 0.0], dtype=np.float32) * 1e3)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.subplots_adjust(
        top=0.933, bottom=0.126, left=0.095, right=0.966, hspace=0.7, wspace=0.0
    )

    if args["tolatex"]:
        plt_config.remove_legends(axes)
        plt_config.adjust_axes(axes)
        g_plt_utils.add_subfig_labels(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath=pl.Path(
                "thesis_figures",
                "results_and_analyses/shell/"
                if args["save_sub_folder"] is None
                else args["save_sub_folder"],
            ),
            file_name="compare_instant_exp_growth_rates"
            if args["save_fig_name"] is None
            else args["save_fig_name"],
        )


def collect_suppl_exp_growth_rate_plots(args):
    # Make axes
    fig, axes = plt.subplots(nrows=1, ncols=1)

    plt_compare.plot_exp_growth_rate_comparison(args, axes=axes)
    axes.set_xlim(-0.0001, 0.005)
    axes.set_ylim(-2000, None)
    axes.set_yticks([-1000, 0, 1000])
    axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if args["tolatex"]:
        plt_config.remove_legends(axes)
        plt_config.adjust_axes(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath=pl.Path(
                "thesis_figures/appendices/extra_plots/" + args["save_sub_folder"]
                if args["save_sub_folder"] is not None
                else ""
            ),
            file_name="compare_instant_exp_growth_rates"
            if args["save_fig_name"] is None
            else args["save_fig_name"],
        )


def collect_error_norm_plots(args):
    # Make axes
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    if cfg.MODEL == cfg.Models.SHELL_MODEL:
        specific_runs_per_profile_dict = {
            "bv": None,
            "bv_eof": [0, 9, 17],
            "sv": [0, 9, 17],
            "lv": [1, 9, 17],
        }
    else:
        specific_runs_per_profile_dict = None

    # Plot BV, BV-EOF, SV, LV
    args["perturbations"] = ["bv", "bv_eof", "sv", "lv"]
    copy_exp_folder = copy.deepcopy(args["exp_folder"])
    twin_axes = plt_compare.plot_error_norm_comparison(
        args,
        axes=axes[0],
        specific_runs_per_profile_dict=specific_runs_per_profile_dict,
    )
    plt_config.hide_axis_labels(twin_axes)

    # Plot RD, NM, RF
    args["perturbations"] = ["rd", "nm", "rf"]
    # Reset exp_folder
    args["exp_folder"] = copy_exp_folder
    twin_axes = plt_compare.plot_error_norm_comparison(args, axes=axes[1])
    plt_config.hide_axis_labels(twin_axes)

    g_plt_utils.add_subfig_labels(axes)

    if cfg.MODEL == cfg.Models.LORENTZ63:
        ylabel_suffix = "$||\\mathbf{x}'||$"
        ylabel_right = "x"
    elif cfg.MODEL == cfg.Models.SHELL_MODEL:
        ylabel_suffix = "$||\\mathbf{u}'||$"
        ylabel_right = "Ref. energy, $\\frac{1}{2} u_n u_n^*$"

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
    label_axes_right.set_ylabel(ylabel_right, labelpad=30)

    plt_config.hide_axis_ticks([label_axes, label_axes_right])

    # Remove spines
    for spine, spine_right_axes in zip(label_axes.spines, label_axes_right.spines):
        label_axes.spines[spine].set_visible(False)
        label_axes_right.spines[spine_right_axes].set_visible(False)

    fig.subplots_adjust(left=0.120, bottom=0.155, right=0.880, top=0.935)

    if args["save_fig"]:
        if cfg.MODEL == cfg.Models.LORENTZ63:
            subfolder = "l63"
        elif cfg.MODEL == cfg.Models.SHELL_MODEL:
            subfolder = "shell"

        g_plt_utils.save_figure(
            args,
            subpath=f"thesis_figures/results_and_analyses/{subfolder}/",
            file_name="compare_error_norm",
        )


def collect_sv_vec_compare_plots(args):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    copy_args1 = copy.deepcopy(args)
    copy_args2 = copy.deepcopy(args)
    copy_args1["vectors"] = ["lv", "alv", "fsv"]
    plt_compare.plt_vec_compared_to_lv(copy_args1, axes=axes[1], pair_vectors=True)

    copy_args2["vectors"] = ["lv", "alv", "sv"]
    plt_compare.plt_vec_compared_to_lv(copy_args2, axes=axes[0], pair_vectors=True)

    # Remove labels
    for ax in axes:
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.supxlabel("$t_{{OPT}}$")
    fig.supylabel("Absolute projectibility")

    fig.subplots_adjust(
        top=0.976, bottom=0.204, left=0.119, right=0.992, hspace=0.2, wspace=0.092
    )

    if args["tolatex"]:
        plt_config.remove_legends(axes)
        plt_config.adjust_axes(axes)
        g_plt_utils.add_subfig_labels(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/pt_methods",
            file_name="projectibility_vs_opt_sv_fsv_vs_lv",
        )


def collect_bv_vec_compare_plots(args):
    fig, axes = plt.subplots(nrows=1, ncols=1)

    args["vectors"] = ["lv", "bv"]
    plt_compare.plt_vec_compared_to_lv(args, axes=axes)

    # fig.subplots_adjust(
    #     top=0.976, bottom=0.204, left=0.119, right=0.992, hspace=0.2, wspace=0.092
    # )

    if args["tolatex"]:
        plt_config.remove_legends(axes)
        plt_config.adjust_axes(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/pt_methods",
            file_name="projectibility_vs_opt_bv_vs_lv",
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
    elif "collect_sv_vec_compare_plots" in args["plot_type"]:
        collect_sv_vec_compare_plots(args)
    elif "collect_bv_vec_compare_plots" in args["plot_type"]:
        collect_bv_vec_compare_plots(args)
    elif "collect_exp_growth_rate_compare_plots" in args["plot_type"]:
        collect_exp_growth_rate_plots(args)
    elif "collect_suppl_exp_growth_rate_compare_plots" in args["plot_type"]:
        collect_suppl_exp_growth_rate_plots(args)
    elif "collect_sv_exp_growth_rate_compare_plots" in args["plot_type"]:
        collect_sv_exp_growth_rate_plots(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args, tight_layout_rect=[0, 0, 0.9, 1])
