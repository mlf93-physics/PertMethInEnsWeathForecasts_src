import sys

sys.path.append("..")
import pathlib as pl
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import general.plotting.plot_config as g_plt_config
import shell_model_experiments.params as sh_params
import shell_model_experiments.plotting.plot_data as sh_plot
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.plotting.plot_data as l63_plot
import general.utils.importing.import_data_funcs as g_import
import general.utils.util_funcs as g_utils
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_data as g_plt_data
import general.analyses.plot_analyses as g_plt_anal
import general.utils.importing.import_perturbation_data as pt_import
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

    for i in range(args["n_units"]):
        # fig, ax1 = plt.subplots(1, 1)
        # cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=n_vectors)
        # ax1.set_prop_cycle("color", cmap_list)
        # ax1.plot(lyapunov_vector_units[i, :, :].T, "--")

        # # Reset color cycle
        # ax1.set_prop_cycle("color", cmap_list)
        # ax1.plot(breed_vector_units[i, :, :].T, "-")
        # ax1.set_title(f"Breed/Lyapunov vectors | unit {i}")

        orthogonality_matrix = (
            breed_vector_units[i, :, :] @ lyapunov_vector_units[i, :, :].T
        )

        fig, ax2 = plt.subplots(1, 1)
        sb.heatmap(
            orthogonality_matrix, cmap="Reds", vmin=-1, vmax=1, annot=True, fmt=".2f"
        )
        ax2.set_xlabel("Lyapunov vector index")
        ax2.set_ylabel("Breed vector index")
        ax2.set_title(f"Orthogonality between Breed/Lyapunov vectors | unit {i}")


if __name__ == "__main__":
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args = compare_plot_arg_parser.args
    print("args", args)

    if "vec_compare" in args["plot_type"]:
        plt_vector_comparison(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    if not args["noplot"]:
        plt.tight_layout()
        plt.show()
