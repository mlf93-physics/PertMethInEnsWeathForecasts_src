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
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.importing.import_utils as g_imp_utils
import general.utils.util_funcs as g_utils
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from mpl_toolkits import mplot3d
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

    real_lines = axes[0].plot(singular_values.T.real)
    colors = [line.get_color() for line in real_lines]
    imag_lines = axes[0].plot(singular_values.T.imag, linestyle="--")

    for i, line in enumerate(imag_lines):
        line.set_color(colors[i])
        real_lines[i].set_label(labels[i])

    # Generate title
    s_value_dist_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Singular value dist "
        + f"{'normalized' if 'normalize' in plot_args else ''}| shell model \n",
    )

    # axes[0].legend()
    axes[0].set_ylabel("Singular value")
    axes[0].set_xlabel("Singular value index")
    axes[0].set_title(s_value_dist_title)

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

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("SV index")
    plt.ylabel("Shell index")


def plot_s_vectors_average(args, axes: plt.Axes = None, plot_args: list = []):

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
        axes = plt.axes()

    mean_abs_singular_vector_units = np.mean(np.abs(singular_vector_units), axis=0)

    sb.heatmap(
        mean_abs_singular_vector_units.T ** 2,
        ax=axes,
    )

    axes.invert_yaxis()
    axes.set_xlabel("SV index")
    axes.set_ylabel("Shell index")

    # Generate title
    s_vector_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Averaged normalized SVs | shell model \n",
    )
    axes.set_title(s_vector_title)


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
        plot_s_values(args, plot_args=["normalize"])
    elif "s_vectors" in args["plot_type"]:
        plot_s_vectors_units(args)
    elif "s_vectors_average" in args["plot_type"]:
        plot_s_vectors_average(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    profiler.stop()
    print(profiler.output_text(color=True))
    g_plt_utils.save_or_show_plot(args)
