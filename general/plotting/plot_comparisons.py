import sys

sys.path.append("..")
import math
import pathlib as pl

import config as cfg
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import lorentz63_experiments.params.params as l63_params
import matplotlib.pyplot as plt
import seaborn as sb
import shell_model_experiments.params as sh_params
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

    breed_vector_units, breed_vec_header_dicts = pt_import.import_perturb_vectors(args)
    # breed_vector_units = np.squeeze(breed_vector_units, axis=0)
    # Normalize vectors
    breed_vector_units = g_utils.normalize_array(
        breed_vector_units, norm_value=1, axis=2
    )

    # Second folder: lyapunov vectors
    args["exp_folder"] = args["exp_folders"][1]
    lyapunov_vector_units, lyapunov_vec_header_dicts = pt_import.import_perturb_vectors(
        args
    )

    # lyapunov_vector_units = np.squeeze(lyapunov_vector_units, axis=0)
    # Normalize vectors
    lyapunov_vector_units = g_utils.normalize_array(
        lyapunov_vector_units, norm_value=1, axis=2
    )

    n_vectors = args["n_runs_per_profile"]

    num_subplot_cols = math.floor(args["n_units"] / 2)
    num_subplot_rows = math.ceil(args["n_units"] / num_subplot_cols)

    _, axes1 = plt.subplots(num_subplot_rows, num_subplot_cols)
    _, axes2 = plt.subplots(num_subplot_rows, num_subplot_cols)
    axes1 = axes1.ravel()
    axes2 = axes2.ravel()

    for i in range(args["n_units"]):
        cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=n_vectors)
        axes1[i].set_prop_cycle("color", cmap_list)
        axes1[i].plot(lyapunov_vector_units[i, :, :].T, "--")

        # Reset color cycle
        axes1[i].set_prop_cycle("color", cmap_list)
        axes1[i].plot(breed_vector_units[i, :, :].T, "-")
        axes1[i].set_title(f"Breed(-)/Lyapunov(--) vectors | unit {i}")

        orthogonality_matrix = (
            breed_vector_units[i, :, :] @ lyapunov_vector_units[i, :, :].T
        )

        sb.heatmap(
            orthogonality_matrix,
            cmap="Reds",
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            ax=axes2[i],
            annot_kws={"fontsize": 8},
        )
        axes2[i].invert_yaxis()
        axes2[i].set_xlabel("Lyapunov vector index")
        axes2[i].set_ylabel("Breed vector index")
        axes2[i].set_title(f"Orthogonality between\nBreed/Lyapunov vectors | unit {i}")


def plot_error_norm_comparison(args: dict):
    """Plots a comparison of the error norm based in several different
    perturbation techniques

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    axes = plt.axes()

    args["endpoint"] = True

    if args["exp_folders"] is not None:
        len_folders = len(args["exp_folders"])
    elif args["exp_folder"] is not None:
        # Get dirs in path
        _path = pl.Path(args["datapath"], args["exp_folder"])
        _dirs = g_utils.get_dirs_in_path(_path)
        len_folders = len(_dirs)

        if len_folders == 0:
            args["exp_folders"] = args["exp_folder"]
        else:
            # Sort out dirs not named *_perturbations
            args["exp_folders"] = [
                str(pl.Path(_dirs[i].parent.name, _dirs[i].name))
                for i in range(len_folders)
                if "perturbations" in _dirs[i].name
            ]

        # Update number of folders after filtering
        len_folders = len(args["exp_folders"])

    cmap_list = g_plt_utils.get_non_repeating_colors(
        n_colors=len_folders, cmap=plt.cm.rainbow
    )

    line_counter = 0
    for i, folder in enumerate(args["exp_folders"]):
        # Set exp_folder
        args["exp_folder"] = folder

        g_plt_data.plot_error_norm_vs_time(
            args,
            axes=axes,
            cmap_list=[cmap_list[i]],
            legend_on=False,
            normalize_start_time=False,
        )
        lines = list(axes.get_lines())
        lines[line_counter].set_label(folder)

        line_counter += len(lines) - line_counter

    plt.legend()


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args: dict = compare_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    if "vec_compare" in args["plot_type"]:
        plt_vector_comparison(args)
    elif "error_norm_compare" in args["plot_type"]:
        plot_error_norm_comparison(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args)
