import sys

sys.path.append("..")
import math
import pathlib as pl

import config as cfg
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.importing.import_utils as g_imp_utils
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.plotting.plot_data as l63_plot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import shell_model_experiments.params as sh_params
import shell_model_experiments.plotting.plot_data as sh_plot
from general.params.model_licences import Models

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = sh_params
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params


def plt_vector_comparison(args):
    """Plot a comparison between breed and lyapunov vectors

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    # Setup number of units
    # args["n_units"] = 1

    # First folder: breed vectors
    args["exp_folder"] = args["exp_folders"][0]

    breed_vector_units, _, _, breed_vec_header_dicts = pt_import.import_perturb_vectors(
        args
    )
    # breed_vector_units = np.squeeze(breed_vector_units, axis=0)
    # Normalize vectors
    breed_vector_units = g_utils.normalize_array(
        breed_vector_units, norm_value=1, axis=2
    )

    # Second folder: lyapunov vectors
    args["exp_folder"] = args["exp_folders"][1]
    (
        lyapunov_vector_units,
        _,
        _,
        lyapunov_vec_header_dicts,
    ) = pt_import.import_perturb_vectors(args)

    # lyapunov_vector_units = np.squeeze(lyapunov_vector_units, axis=0)
    # Normalize vectors
    lyapunov_vector_units = g_utils.normalize_array(
        lyapunov_vector_units, norm_value=1, axis=2
    )

    n_vectors = args["n_runs_per_profile"]

    num_subplot_cols = math.floor(args["n_profiles"] / 2)
    num_subplot_rows = math.ceil(args["n_profiles"] / num_subplot_cols)

    fig1, axes1 = plt.subplots(
        num_subplot_rows, num_subplot_cols, sharex=True, sharey=True
    )
    fig2, axes2 = plt.subplots(
        num_subplot_rows, num_subplot_cols, sharex=True, sharey=True
    )
    axes1 = axes1.ravel()
    axes2 = axes2.ravel()

    # Add cbar axes
    cbar_ax = fig2.add_axes([0.91, 0.3, 0.03, 0.4])

    for i in range(args["n_profiles"]):
        cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=n_vectors)
        axes1[i].set_prop_cycle("color", cmap_list)
        axes1[i].plot(lyapunov_vector_units[i, :, :].T.real, "--")

        # Reset color cycle
        axes1[i].set_prop_cycle("color", cmap_list)
        axes1[i].plot(breed_vector_units[i, :, :].T.real, "-")
        axes1[i].set_title(f"unit {i}")

        orthogonality_matrix = np.abs(
            breed_vector_units[i, :, :].conj() @ lyapunov_vector_units[i, :, :].T
        )

        sb.heatmap(
            orthogonality_matrix,
            cmap="Reds",
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            ax=axes2[i],
            cbar=i == 0,
            cbar_ax=None if i else cbar_ax,
            annot_kws={"fontsize": 8},
        )
        axes2[i].invert_yaxis()
        axes2[i].set_xlabel("Lyapunov vector index")
        axes2[i].set_ylabel("Breed vector index")
        axes2[i].set_title(f"unit {i}")

    fig1.suptitle(f"Breed(-)/Lyapunov(--) vectors")
    fig2.suptitle(f"Orthogonality between\nBreed/Lyapunov vectors")
    fig2.tight_layout(rect=[0, 0, 0.9, 1])


def plot_error_norm_comparison(args: dict):
    """Plots a comparison of the error norm based in several different
    perturbation techniques

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    args["endpoint"] = True

    if args["exp_folders"] is not None:
        len_folders = len(args["exp_folders"])
    elif args["exp_folder"] is not None:
        # Get dirs in path
        _path = pl.Path(args["datapath"], args["exp_folder"])
        _dirs = g_utils.get_dirs_in_path(_path)
        len_folders = len(_dirs)

        if len_folders == 0:
            args["exp_folders"] = [args["exp_folder"]]
        else:
            # Sort out dirs not named *_perturbations
            args["exp_folders"] = [
                str(pl.Path(_dirs[i].parent.name, _dirs[i].name))
                for i in range(len_folders)
                if "perturbations" in _dirs[i].name
            ]

        # Update number of folders after filtering
        len_folders = len(args["exp_folders"])

    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    line_counter = 0
    for i, folder in enumerate(args["exp_folders"]):
        # Set exp_folder
        args["exp_folder"] = folder

        g_plt_data.plot_error_norm_vs_time(
            args,
            axes=axes[0],
            cmap_list=[cmap_list[i]],
            legend_on=False,
            normalize_start_time=False,
            plot_args=[],
        )
        lines: list = list(axes[0].get_lines())
        lines[line_counter].set_label(folder)

        len_lines = len(lines)
        line_counter += len_lines - line_counter

        # for j in range(0, len_lines, 3):
        #     lines[j].set_linestyle("-")
        #     lines[(j + 1) % len_lines].set_linestyle("--")
        #     lines[(j + 2) % len_lines].set_linestyle("-.")

    axes[0].legend()

    # Import perturbation and experiment info files of last perturbation folder
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["datapath"], args["exp_folder"])
    )
    exp_setup = g_import.import_exp_info_file(args)

    start_time, end_time = g_imp_utils.get_start_end_times_from_exp_setup(
        exp_setup, pert_info_dict
    )

    args["ref_start_time"] = start_time
    args["ref_end_time"] = end_time

    if cfg.MODEL == Models.SHELL_MODEL:
        sh_plot.plots_related_to_energy(
            args,
            axes=axes[1],
            plot_args=[],
        )
    elif cfg.MODEL == Models.LORENTZ63:
        l63_plot.plot_energy(args, axes=axes[1])


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args: dict = compare_plot_arg_parser.args

    # Initiate variables
    params.initiate_sdim_arrays(args["sdim"])

    g_ui.confirm_run_setup(args)

    if "vec_compare" in args["plot_type"]:
        plt_vector_comparison(args)
    elif "error_norm_compare" in args["plot_type"]:
        plot_error_norm_comparison(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args)
