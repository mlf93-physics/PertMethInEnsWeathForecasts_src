"""Plotting functions relevant for the perturbations of the Lorentz63 model experiments

Example
-------
python plotting/plot_perturbations.py --plot_type=<>

"""

import sys

sys.path.append("..")
import re
import pathlib as pl
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from lorentz63_experiments.params.params import *
import lorentz63_experiments.analyses.normal_mode_analysis as nm_analysis
import lorentz63_experiments.utils.util_funcs as l_utils
import general.utils.arg_utils as a_utils
import general.utils.running.runner_utils as r_utils
import general.utils.experiments.exp_utils as e_utils
import general.utils.importing.import_data_funcs as g_import
import general.plotting.plot_data as g_plt_data
import general.utils.plot_utils as g_plt_utils
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
import sklearn.cluster as skl_cluster
import config as cfg

cfg.GLOBAL_PARAMS.record_max_time = 3000


def plot_pert_vectors(args, axes: plt.Axes = None):

    if axes is None:
        axes = plt.axes()

    e_utils.update_compare_exp_folders(args)

    n_runs_per_profile_dict = {
        "rd": 500,
        "nm": 500,
        "bv": 2,
        "bv_eof": 2,
        "sv": 2,
        "rf": 10,
    }

    vector_folders = [folder for folder in args["exp_folders"] if "vectors" in folder]

    # Import perturb vectors and plot
    for i, folder in enumerate(vector_folders):
        raw_perturbations = True

        args["exp_folder"] = folder
        # Update arguments
        args["pert_mode"] = args["vectors"][i]
        args["n_runs_per_profile"] = n_runs_per_profile_dict[args["vectors"][i]]

        perturbations, perturb_positions, _ = r_utils.prepare_perturbations(
            args, raw_perturbations=raw_perturbations
        )

        axes.scatter(
            perturbations[0, :],
            perturbations[1, :],
            label=args["vectors"][i],
            zorder=10,
        )

    # Filter out already plotted perturbations
    filtered_perturbations = [
        item for item in args["perturbations"] if item not in args["vectors"]
    ]
    # Generate all other perturbation vectors
    for i, mode in enumerate(filtered_perturbations):
        args["pert_mode"] = mode

        args["n_runs_per_profile"] = n_runs_per_profile_dict[mode]
        args["start_times"] = [perturb_positions[0] * stt]

        perturbations, perturb_positions, _ = r_utils.prepare_perturbations(
            args, raw_perturbations=raw_perturbations
        )

        axes.scatter(
            perturbations[0, :],
            perturbations[1, :],
            label=mode,
            alpha=0.6 if mode in ["rd", "nm"] else 1.0,
            zorder=0,
        )

    title = g_plt_utils.generate_title(
        args, title_header="Perturbation Vectors", detailed=False
    )

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title(title)
    axes.legend()


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args: dict = compare_plot_arg_parser.args

    a_utils.react_on_comparison_arguments(args)

    # Initiate arrays
    # initiate_sdim_arrays(args["sdim"])
    g_ui.confirm_run_setup(args)

    if "time_to_run" in args:
        args["Nt"] = int(args["time_to_run"] / dt * sample_rate)

    if "pert_vectors" in args["plot_type"]:
        plot_pert_vectors(args)
    else:
        raise ValueError(f"No plot method present for plot_type={args['plot_type']}")

    g_plt_utils.save_or_show_plot(args)
