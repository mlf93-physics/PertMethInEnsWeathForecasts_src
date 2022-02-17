"""Make plots related to the singular vector calculations

Example
-------
python ../general/plotting/singular_vector_plotter.py
--plot_type=s_values
--exp_folder=pt_vectors/test1_after_params_refactor
--endpoint

"""

import sys

sys.path.append("..")
import pathlib as pl
import math
import config as cfg
import general.analyses.breed_vector_eof_analysis as bv_analysis
import general.analyses.plot_analyses as g_plt_anal
import general.analyses.analyse_data as g_anal
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.importing.import_utils as g_imp_utils
import general.utils.util_funcs as g_utils
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
from mpl_toolkits import mplot3d
import seaborn as sb
import numpy as np
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from pyinstrument import Profiler

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.plotting.plot_data as sh_plot
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType
    import shell_model_experiments.utils.special_params as sh_sparams

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.plotting.plot_data as l63_plot
    import lorentz63_experiments.params.special_params as l63_sparams

    params = l63_params
    sparams = l63_sparams


def plot_s_values(args, axes: plt.Axes = None, plot_args=[]):

    # Prepare axes
    if axes is None:
        fig, axes = plt.subplots(ncols=1, nrows=2)

    # Import breed vectors
    (
        _,
        singular_values,
        _,
        _,
        header_dicts,
    ) = pt_import.import_perturb_vectors(args, raw_perturbations=True)

    if "normalize" in plot_args:
        singular_values = g_utils.normalize_array(singular_values, norm_value=1, axis=1)

    labels = []
    for i, header_dict in enumerate(header_dicts):
        labels.append(f"Start time: {header_dict['val_pos']*params.stt:.2f}")

    real_lines = axes[0].plot(singular_values.T.real, "k", alpha=0.3)
    colors = [line.get_color() for line in real_lines]
    imag_lines = axes[0].plot(singular_values.T.imag, linestyle="--", alpha=0.3)

    for i, line in enumerate(imag_lines):
        line.set_color(colors[i])
        real_lines[i].set_label(labels[i])

    # Generate title
    s_value_dist_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Singular value dist "
        + f"{'normalized' if 'normalize' in plot_args else ''}\n",
    )

    # axes[0].legend()
    axes[0].set_ylabel("Singular value")
    axes[0].set_xlabel("Singular value index")
    axes[0].set_title(s_value_dist_title)
    if "normalize" in plot_args:
        axes[0].set_ylim(-0.2, 0.4)
    else:
        axes[0].set_yscale("log")

    # Get val pos from dicts
    val_pos_list = g_utils.get_values_from_dicts(header_dicts, "val_pos")

    # Set ref start and end times
    args["ref_start_time"] = np.min(val_pos_list) * params.stt
    args["ref_end_time"] = np.max(val_pos_list) * params.stt

    g_plt_data.plot_energy(
        args, axes=axes[1], plot_args=[], plot_kwargs={"exp_file_type": "vectors"}
    )


def plot_s_vectors_units(args, axes: plt.Axes = None, plot_args: list = []):

    # Import breed vectors
    (
        singular_vector_units,
        singular_values,
        _,
        _,
        header_dicts,
    ) = pt_import.import_perturb_vectors(
        args, raw_perturbations=True, dtype=sparams.dtype
    )

    # Normalize
    singular_vector_units = g_utils.normalize_array(
        singular_vector_units, norm_value=1, axis=2
    )

    n_units = singular_vector_units.shape[0]
    # Prepare axes
    if axes is None:
        num_subplot_cols = math.floor(n_units / 2) + 1
        num_subplot_rows = math.ceil(n_units / num_subplot_cols)
        fig, axes = plt.subplots(
            ncols=num_subplot_cols,
            nrows=num_subplot_rows,
        )

    axes = axes.ravel()

    for unit in range(n_units):
        sb.heatmap(
            np.abs(singular_vector_units[unit, :, :].T),
            ax=axes[unit],
        )
        axes[unit].set_title(
            f"Start time: {header_dicts[unit]['val_pos']*params.stt:.2f}"
        )
        axes[unit].invert_yaxis()

    # Generate title
    s_vector_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Normalized SV units \n",
    )
    fig.suptitle(s_vector_title)
    fig.supxlabel("SV index")
    fig.supylabel("Shell index")


def plot_s_vectors_average(args, axes: plt.Axes = None):

    # Prepare plot_kwargs
    plot_kwargs: dict = {
        "xlabel": "SV index",
        "ylabel": "Shell index",
        "title_header": "Averaged SVs",
    }

    # Import breed vectors
    g_plt_data.plot2D_average_vectors(args, axes=axes, plot_kwargs=plot_kwargs)


def plot3D_s_vectors_average(args, axes: plt.Axes = None, plot_args: list = []):

    # Import breed vectors
    (
        singular_vector_units,
        singular_values,
        _,
        _,
        header_dicts,
    ) = pt_import.import_perturb_vectors(
        args, raw_perturbations=True, dtype=sparams.dtype
    )

    # Normalize
    singular_vector_units = g_utils.normalize_array(
        singular_vector_units, norm_value=1, axis=2
    )

    # Prepare axes
    if axes is None:
        # axes = plt.axes()
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    mean_abs_singular_vector_units = np.mean(np.abs(singular_vector_units), axis=0)

    # Prepare plot settings
    mpl_ticker.MaxNLocator.default_params["integer"] = True

    # Make data
    shells = np.arange(0, params.sdim, 1)
    sv_index = np.arange(0, params.sdim, 1)
    sv_index, shells = np.meshgrid(sv_index, shells)
    surf_plot = axes.plot_surface(
        sv_index, shells, mean_abs_singular_vector_units.T, cmap="Reds"
    )
    axes.set_xlim(params.sdim, 0)
    axes.set_ylim(0, params.sdim)
    fig.colorbar(surf_plot, ax=axes, pad=0.1)
    surf_plot.set_clim(0, 0.5)

    axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
    axes.yaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    axes.set_xlabel("SV index, j")
    axes.set_ylabel("Shell index, i")
    axes.set_zlabel("$\\langle|sv_i^j|\\rangle$")
    axes.view_init(elev=28.0, azim=-60)
    axes.set_zlim(0, 0.5)

    # Generate title
    s_vector_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Averaged normalized SVs \n",
    )
    axes.set_title(s_vector_title)


def plot_s_vector_ortho(args, axes=None):

    # Import breed vectors
    (
        singular_vector_units,
        singular_values,
        _,
        _,
        header_dicts,
    ) = pt_import.import_perturb_vectors(
        args, raw_perturbations=True, dtype=sparams.dtype
    )

    # Normalize
    singular_vector_units = g_utils.normalize_array(
        singular_vector_units, norm_value=1, axis=2
    )

    # Prepare axes
    if axes is None:
        num_subplot_cols = math.floor(args["n_profiles"] / 2)
        num_subplot_rows = math.ceil(args["n_profiles"] / num_subplot_cols)
        fig, axes = plt.subplots(
            ncols=num_subplot_cols, nrows=num_subplot_rows, sharey=True, sharex=True
        )

    try:
        axes.size
    except AttributeError:
        axes = [axes]
    else:
        axes = axes.ravel()

    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    for i in range(args["n_profiles"]):
        orthogonality_matrix = g_plt_anal.orthogonality_of_vectors(
            singular_vector_units[i, :, :]
        )

        sb.heatmap(
            orthogonality_matrix,
            ax=axes[i],
            cmap="Reds",
            vmin=0,
            vmax=1,
            # annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 8},
            cbar=i == 0,
            cbar_ax=None if i else cbar_ax,
            cbar_kws=dict(use_gridspec=True, label="Orthogonality"),
        )

        axes[i].set_title(f'Val. pos. {header_dicts[i]["val_pos"]*params.stt:.2f}')

    # Generate title
    s_vector_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Orthogonality between SVs \n",
    )
    fig.suptitle(s_vector_title)
    fig.supxlabel("SV index")
    fig.supylabel("SV index")
    fig.tight_layout(rect=[0, 0, 0.9, 1])


def plot_s_vector_ortho_average(args, axes=None):

    # Import breed vectors
    (
        singular_vector_units,
        singular_values,
        _,
        _,
        header_dicts,
    ) = pt_import.import_perturb_vectors(
        args, raw_perturbations=True, dtype=np.complex128
    )

    # Normalize
    singular_vector_units = g_utils.normalize_array(
        singular_vector_units, norm_value=1, axis=2
    )

    # Prepare axes
    if axes is None:
        fig, axes = plt.subplots(ncols=1, nrows=1)

    orthogonality_matrix = 0
    for i in range(args["n_profiles"]):
        orthogonality_matrix += g_plt_anal.orthogonality_of_vectors(
            singular_vector_units[i, :, :]
        )

    orthogonality_matrix /= args["n_profiles"]

    sb.heatmap(
        orthogonality_matrix,
        ax=axes,
        cmap="Reds",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8},
        cbar_kws=dict(use_gridspec=True, label="Orthogonality"),
    )

    axes.set_title(f'Val. pos. {header_dicts[i]["val_pos"]*params.stt:.2f}')

    # Generate title
    s_vector_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Averaged orthogonality between SVs \n",
    )
    fig.suptitle(s_vector_title)
    fig.supxlabel("SV index")
    fig.supylabel("SV index")
    fig.tight_layout(rect=[0, 0, 0.9, 1])


def plot_sv_error_norm(args):

    axes = plt.axes()

    g_plt_data.plot_error_norm_vs_time(
        args=args,
        legend_on=True,
        axes=axes,
        plot_args=["detailed_title"],
        normalize_start_time=False,
    )


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

    # Make profiler
    profiler = Profiler()
    profiler.start()

    if "s_values" in args["plot_type"]:
        plot_s_values(args, plot_args=[])
    elif "s_vectors" in args["plot_type"]:
        plot_s_vectors_units(args)
    elif "s_vectors_average" in args["plot_type"]:
        plot_s_vectors_average(args)
    elif "s_vectors_average_3D" in args["plot_type"]:
        plot3D_s_vectors_average(args)
    elif "s_vector_ortho" in args["plot_type"]:
        plot_s_vector_ortho(args)
    elif "s_vector_ortho_average" in args["plot_type"]:
        plot_s_vector_ortho_average(args)
    elif "sv_error_norm" in args["plot_type"]:
        plot_sv_error_norm(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    profiler.stop()
    print(profiler.output_text(color=True))
    g_plt_utils.save_or_show_plot(args)
