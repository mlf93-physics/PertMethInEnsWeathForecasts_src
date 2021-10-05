import sys

sys.path.append("..")
import argparse
import math
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler
import general.plotting.plot_config as g_plt_config
import shell_model_experiments.params as sh_params
import shell_model_experiments.plotting.plot_data as sh_plot
import shell_model_experiments.plotting.plot_data as pl_data
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.plotting.plot_data as l63_plot
import general.analyses.lorentz_block_analysis as lr_analysis
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_data as g_plt_data
import general.plotting.plot_params as plt_params
import general.utils.argument_parsers as a_parsers
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params

# Setup plotting defaults
g_plt_config.setup_plotting_defaults()


def plot_tlm_solution(args):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    exp_setup = g_import.import_exp_info_file(args)

    g_plt_data.plot_error_norm_vs_time(
        args=args, normalize_start_time=False, axes=axes[0]
    )

    # Prepare ref import
    start_time = exp_setup["start_times"][0] if "start_times" in exp_setup else 0
    args["ref_start_time"] = start_time
    args["ref_end_time"] = start_time + 6 * exp_setup["vector_offset"]

    if MODEL == Models.SHELL_MODEL:
        sh_plot.plots_related_to_energy(args, axes=axes[1])
    elif MODEL == Models.LORENTZ63:
        l63_plot.plot_energy(args, axes=axes[1])


if __name__ == "__main__":
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    args = stand_plot_arg_parser.args
    print("args", args)

    if "tlm_error_norm" in args["plot_type"]:
        plot_tlm_solution(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    if not args["noplot"]:
        plt.tight_layout()
        plt.show()
