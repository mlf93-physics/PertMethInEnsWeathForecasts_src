"""Make plots related to the calculation of the Lyapunov vectors

Example
-------
python ../general/plotting/lyapunov_vector_plotter.py
--plot_type=tlm_error_norm
--exp_folder=pt_vectors/debug_temp1
--endpoint
"""

import sys

sys.path.append("..")
import copy
import pathlib as pl

import config as cfg
import general.analyses.plot_analyses as g_plt_anal
import general.plotting.plot_config as g_plt_config
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import general.plotting.plot_config as plt_config
import general.utils.importing.import_perturbation_data as pt_import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from general.params.model_licences import Models

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.plotting.plot_data as sh_plot
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.plotting.plot_data as l63_plot

    params = l63_params

# Setup plotting defaults
# g_plt_config.setup_plotting_defaults()


def plot_lyapunov_vectors_average(args: dict, axes: plt.Axes = None):
    # Prepare plot_kwargs
    plot_kwargs: dict = {
        "xlabel": "$i$",
        "ylabel": "$n$",
        "title_header": "Averaged LVs",
        "vector_label": "$\\langle|\\xi^{\\infty}_{n,i}| \\rangle$",
    }

    g_plt_data.plot2D_average_vectors(
        args,
        axes=axes,
        plot_kwargs=plot_kwargs,
        characteristic_value_name="$\\lambda_i/t_{{OPT}}$",
        no_char_values=False,
    )

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/" + args["save_sub_folder"],
            file_name="average_lv_vectors_with_exponents",
        )


def plot_adj_lyapunov_vectors_average(args: dict, axes: plt.Axes = None):
    # Prepare plot_kwargs
    plot_kwargs: dict = {
        "xlabel": "$i$",
        "ylabel": "$n$",
        "title_header": "Averaged adjoint LVs",
        "vector_label": "$\\langle|\\xi^{-\\infty}_{n,i}| \\rangle$",
    }

    g_plt_data.plot2D_average_vectors(
        args,
        axes=axes,
        plot_kwargs=plot_kwargs,
        characteristic_value_name="$\\lambda_i/t_{{OPT}}$",
        no_char_values=True,
    )

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/" + args["save_sub_folder"],
            file_name="average_alv_vectors",
        )


def plot_tlm_solution(args, axes=None):
    if axes is None:
        _, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    elif len(axes) != 2:
        raise ValueError(
            "Wrong number of axes given to the plot function. "
            + f"2 is needed; {len(axes)}"
        )

    # Import exp and perturb info files
    exp_setup = g_import.import_exp_info_file(args)
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["datapath"], args["exp_folder"], "perturb_data")
    )

    args["exp_folder"] = pl.Path(exp_setup["folder_name"], exp_setup["sub_exp_folder"])

    g_plt_data.plot_error_norm_vs_time(
        args=args,
        normalize_start_time=False,
        axes=axes[0],
        exp_setup=exp_setup,
        legend_on=False,
    )

    # Prepare ref import
    if "start_times" in exp_setup:
        start_time = exp_setup["start_times"][0]
    elif "eval_times" in exp_setup:
        start_time = exp_setup["eval_times"][0] - exp_setup["integration_time"]
    else:
        raise ValueError("start_time could not be determined from exp setup")

    args["ref_start_time"] = start_time
    args["ref_end_time"] = (
        start_time + pert_info_dict["n_units"] * exp_setup["vector_offset"]
    )

    if cfg.MODEL == Models.SHELL_MODEL:
        sh_plot.plot_energy(args, axes=axes[1])
    elif cfg.MODEL == Models.LORENTZ63:
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

    # Import exp and pert setup
    exp_setup = g_import.import_exp_info_file(args)
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["datapath"], args["exp_folder"], "perturb_data")
    )
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
    n_time_steps = int(pert_info_dict["n_units"] * n_time_steps_per_unit)
    # Prepare time array
    start_time = exp_setup["start_times"][0] if "start_times" in exp_setup else 0
    time_array = (
        np.arange(0, pert_info_dict["n_units"]) * exp_setup["integration_time"]
        + start_time
    )

    # Initialise collection of orthogonality measures
    orthogonality_collection = np.zeros((n_entries, n_time_steps))

    # Run through all units
    for i in range(int(pert_info_dict["n_units"])):
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
            orthogonality_matrix = g_plt_anal.orthogonality_of_vectors(norm_tlm_vectors)

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


def plot_zeroth_lyapunov_vector(args):

    (
        vector_units,
        characteristic_values,
        _,
        _,
        header_dicts,
    ) = pt_import.import_perturb_vectors(
        args, raw_perturbations=True, dtype=np.complex128
    )

    sort_index = np.argsort(characteristic_values, axis=1)[:, ::-1]
    for i in range(vector_units.shape[0]):
        vector_units[i, :, :] = vector_units[i, sort_index[i, :], :]
        characteristic_values[i, :] = characteristic_values[i, sort_index[i, :]]

    vector_units = g_utils.normalize_array(vector_units, norm_value=1, axis=2)
    mean_abs_vector_units = np.mean(np.abs(vector_units), axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.log2(params.k_vec_temp), mean_abs_vector_units[0, :], label="$LV^1$")
    ax.set_yscale("log")
    ax.plot(
        np.log2(params.k_vec_temp),
        params.k_vec_temp ** (-1 / 3),
        "k--",
        label="$k^{{-1/3}}$",
    )
    ax.legend()


if __name__ == "__main__":
    cfg.init_licence()
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

    plt_config.adjust_default_fig_axes_settings(args)

    if "tlm_error_norm" in args["plot_type"]:
        plot_tlm_solution(args)
    elif "tlm_orth_vs_time" in args["plot_type"]:
        plot_tlm_solution_orthogonality_vs_time(args)
    elif "tlm_error_norm_and_orth" in args["plot_type"]:
        plot_tlm_solution_and_orthogonality(args)
    elif "lv_average" in args["plot_type"]:
        plot_lyapunov_vectors_average(args)
    elif "alv_average" in args["plot_type"]:
        plot_adj_lyapunov_vectors_average(args)
    elif "zeroth_lv" in args["plot_type"]:
        plot_zeroth_lyapunov_vector(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args)
