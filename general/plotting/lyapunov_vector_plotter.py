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
import general.plotting.plot_data as g_plt_data
import general.analyses.plot_analyses as g_plt_anal
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


def plot_tlm_solution(args, axes=None):
    if axes is None:
        _, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    elif len(axes) != 2:
        raise ValueError(
            "Wrong number of axes given to the plot function. "
            + f"2 is needed; {len(axes)}"
        )

    exp_setup = g_import.import_exp_info_file(args)

    args["exp_folder"] = pl.Path(exp_setup["folder_name"], exp_setup["sub_exp_folder"])
    g_plt_data.plot_error_norm_vs_time(
        args=args, normalize_start_time=False, axes=axes[0], exp_setup=exp_setup
    )

    # Prepare ref import
    if "start_times" in exp_setup:
        start_time = exp_setup["start_times"][0]
    elif "eval_times" in exp_setup:
        start_time = exp_setup["eval_times"][0] - exp_setup["integration_time"]
    else:
        raise ValueError("start_time could not be determined from exp setup")

    args["ref_start_time"] = start_time
    args["ref_end_time"] = start_time + 6 * exp_setup["vector_offset"]

    if MODEL == Models.SHELL_MODEL:
        sh_plot.plots_related_to_energy(args, axes=axes[1])
    elif MODEL == Models.LORENTZ63:
        l63_plot.plot_energy(args, axes=axes[1])

    axes[0].set_xlim(args["ref_start_time"], args["ref_end_time"])
    axes[1].set_xlim(args["ref_start_time"], args["ref_end_time"])


def plot_tlm_solution_orthogonality_vs_time(args, axes=None):
    """Plot the orthogonality of the TL model solutions vs time for all combinations.
    Since the end vector of a TL model solution (started from random init conditions),
    is the leading local Lyapunov vector (leading LLV), different random init conditions,
    should converge to the same leading LLV.

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    # Setup axes
    if axes is None:
        axes = plt.axes()

    # Import exp setup
    exp_setup = g_import.import_exp_info_file(args)
    # Set exp folder to a subfolder
    args["exp_folder"] = pl.Path(exp_setup["folder_name"], exp_setup["sub_exp_folder"])

    # Import perturbation data
    (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        header_dicts,
        u_ref_stores,
    ) = g_import.import_perturbation_velocities(args, search_pattern="*perturb*.csv")

    # Calculate the number of entries in the upper triangular matrix (excluding diagonal)
    n_entries = int(exp_setup["n_vectors"] * (exp_setup["n_vectors"] - 1) / 2)
    # The number of time steps per unit
    n_time_steps_per_unit = int(round(exp_setup["integration_time"] * params.tts, 0))
    # The total number of time steps for all units
    n_time_steps = int(header_dicts[0]["n_units"] * n_time_steps_per_unit)
    # Prepare time array
    start_time = exp_setup["start_times"][0] if "start_times" in exp_setup else 0
    time_array = (
        np.arange(0, header_dicts[0]["n_units"]) * exp_setup["integration_time"]
        + start_time
    )

    # Initialise collection of orthogonality measures
    orthogonality_collection = np.zeros((n_entries, n_time_steps))

    # Run through all units
    for i in range(int(header_dict["n_units"])):
        # Get TL model solutions of current unit
        tlm_vectors = np.array(
            u_stores[i * exp_setup["n_vectors"] : (i + 1) * exp_setup["n_vectors"]]
        )
        # Run through all time steps for the given unit
        for j in range(n_time_steps_per_unit):
            # Get the TL model vector at the current time step
            tlm_vectors_time_specific = tlm_vectors[:, j, :]

            # Normalize
            norm_tlm_vectors = tlm_vectors_time_specific / np.reshape(
                np.linalg.norm(tlm_vectors_time_specific, axis=1),
                (exp_setup["n_vectors"], 1),
            )
            # Calculate orthonormality
            orthogonality_matrix = g_plt_anal.orthogonality_of_vectors(
                norm_tlm_vectors, args
            )

            # Save orthogonality measure for plotting
            orthogonality_collection[
                :, j + i * n_time_steps_per_unit
            ] = orthogonality_matrix[np.triu_indices(exp_setup["n_vectors"], k=1)]

    orth_plot = sb.heatmap(
        orthogonality_collection,
        xticklabels=n_time_steps_per_unit,
        cmap="Reds",
        ax=axes,
        cbar_kws=dict(
            use_gridspec=True, location="bottom", pad=0.25, label="Orthogonality"
        ),
    )
    orth_plot.set_xticklabels(time_array)
    axes.set_xlabel("Time")
    axes.set_ylabel("Vector pair index")


def plot_tlm_solution_and_orthogonality(args):
    # Setup axes
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False)
    plot_tlm_solution(copy.deepcopy(args), axes[:2])
    plot_tlm_solution_orthogonality_vs_time(copy.deepcopy(args), axes[2])

    fig.tight_layout()


if __name__ == "__main__":
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    args = stand_plot_arg_parser.args
    print("args", args)

    if "tlm_error_norm" in args["plot_type"]:
        plot_tlm_solution(args)
    elif "tlm_orth_vs_time" in args["plot_type"]:
        plot_tlm_solution_orthogonality_vs_time(args)
    elif "tlm_error_norm_and_orth" in args["plot_type"]:
        plot_tlm_solution_and_orthogonality(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    if not args["noplot"]:
        plt.tight_layout()
        plt.show()
