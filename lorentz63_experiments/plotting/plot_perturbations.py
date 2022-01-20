"""Plotting functions relevant for the perturbations of the Lorentz63 model experiments

Example
-------
python plotting/plot_perturbations.py --plot_type=<>

"""

import sys

sys.path.append("..")
import pathlib as pl
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from lorentz63_experiments.params.params import *
import lorentz63_experiments.analyses.normal_mode_analysis as nm_analysis
import lorentz63_experiments.utils.util_funcs as l_utils
import general.utils.running.runner_utils as r_utils
import general.utils.perturb_utils as pt_utils
import general.utils.importing.import_data_funcs as g_import
import general.plotting.plot_data as g_plt_data
import general.utils.plot_utils as g_plt_utils
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
import sklearn.cluster as skl_cluster
import config as cfg

cfg.GLOBAL_PARAMS.record_max_time = 3000


def plot_nm_pert_vectors(args):

    args["pert_mode"] = "nm"

    perturbations, _, _ = r_utils.prepare_perturbations(args, raw_perturbations=True)

    plt.scatter(perturbations[0, :], perturbations[1, :])


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args

    # Initiate arrays
    # initiate_sdim_arrays(args["sdim"])
    g_ui.confirm_run_setup(args)

    # if "time_to_run" in args:
    #     args["Nt"] = int(args["time_to_run"] / dt * sample_rate)

    if "nm_pert_vectors" in args["plot_type"]:
        plot_nm_pert_vectors(args)
    else:
        raise ValueError(f"No plot method present for plot_type={args['plot_type']}")

    g_plt_utils.save_or_show_plot(args)
