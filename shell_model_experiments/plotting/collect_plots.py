"""Make plots related to the standard shell model

Example
-------
python plotting/plot_data.py
--plot_type=error_norm
--exp_folder=temp1_test_no_hyper_diffusivity_ny_n16
--shell_cutoff=12
--endpoint
"""

import sys

sys.path.append("..")
from pathlib import Path
from typing import List

import config as cfg
import general.plotting.plot_data as g_plt_data
from general.utils.module_import.type_import import *
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_config as plt_config
import general.utils.user_interface as g_ui
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import shell_model_experiments.utils.util_funcs as sh_utils
import shell_model_experiments.analyses.analyse_data as sh_analysis
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
from mpl_toolkits import mplot3d
from shell_model_experiments.params.params import ParamsStructType
from shell_model_experiments.params.params import PAR
import plot_data as sh_plot_data


def make_spec_energy_howmoller_plot(args):

    # fig, axes = plt.subplots(nrows=3, ncols=1, gridspec_kw={"height_ratios": [1, 1, 2]})
    fig = plt.figure()
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax2)
    axs = [ax1, ax2, ax3]

    args["ref_end_time"] = 100
    sh_plot_data.plot_energy_spectrum(args=args, axes=ax1)

    args["ref_end_time"] = 20
    sh_plot_data.plot_energy(args, axes=ax2, plot_args=[])

    sh_plot_data.plot_howmoller_diagram_u_energy(args, axes=ax3, fig=fig)

    labels = ["(a)", "(b)", "(c)"]
    for ax, label in zip(axs, labels):
        ax.set_title(label, loc="center", fontsize="medium")

    if args["save_fig"]:
        g_plt_utils.save_figure(
            subpath="thesis_figures/models/",
            file_name="sh_spec_energy_howmoller",
        )


def make_lyapunov_anal_plot(args):
    fig, axes = plt.subplots(
        ncols=1, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    sh_plot_data.plot_eigen_value_dist(args=args, axes=axes[0])
    sh_plot_data.plot_2D_eigen_mode_analysis(args=args, axes=axes[1], fig=fig)


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args
    # Initiate arrays
    # initiate_PAR.sdim_arrays(args["PAR.sdim"])
    # Initiate and update variables and arrays
    sh_utils.update_dependent_params(PAR)
    sh_utils.update_arrays(PAR)

    g_ui.confirm_run_setup(args)

    plt_config.adjust_default_fig_axes_settings(args)

    if "time_to_run" in args:
        args["Nt"] = int(args["time_to_run"] / PAR.dt * PAR.sample_rate)

    if "spec_energy_howmoller" in args["plot_type"]:
        make_spec_energy_howmoller_plot(args)
    if "lyapunov_anal" in args["plot_type"]:
        make_lyapunov_anal_plot(args)

    g_plt_utils.save_or_show_plot(args)
