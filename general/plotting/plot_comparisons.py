"""Make plots to compare the different perturbation methods

Example
-------
python ../general/plotting/plot_comparisons.py
--plot_type=error_norm_compare
--exp_folder=test1_new_params

"""

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
from general.plotting.plot_params import *
from general.utils.module_import.type_import import *
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
from libs.libutils import file_utils as lib_file_utils, type_utils as lib_type_utils
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

    (
        breed_vector_units,
        _,
        _,
        _,
        breed_vec_header_dicts,
    ) = pt_import.import_perturb_vectors(args, raw_perturbations=True)
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
        _,
        lyapunov_vec_header_dicts,
    ) = pt_import.import_perturb_vectors(args, raw_perturbations=True)

    # lyapunov_vector_units = np.squeeze(lyapunov_vector_units, axis=0)
    # Normalize vectors
    lyapunov_vector_units = g_utils.normalize_array(
        lyapunov_vector_units, norm_value=1, axis=2
    )

    n_vectors = args["n_runs_per_profile"]

    if args["n_profiles"] > 1:
        num_subplot_cols = math.floor(args["n_profiles"] / 2)
        num_subplot_rows = math.ceil(args["n_profiles"] / num_subplot_cols)
    else:
        num_subplot_cols = 1
        num_subplot_rows = 1

    fig1, axes1 = plt.subplots(
        num_subplot_rows, num_subplot_cols, sharex=True, sharey=True
    )
    fig2, axes2 = plt.subplots(
        num_subplot_rows, num_subplot_cols, sharex=True, sharey=True
    )

    if args["n_profiles"] == 1:
        axes1 = np.array(axes1)
        axes2 = np.array(axes2)

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

        # print(
        #     "orthogonality_matrix mean",
        #     np.mean(
        #         orthogonality_matrix[~np.eye(orthogonality_matrix.shape[0], dtype=bool)]
        #     ),
        # )

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
        _dirs = lib_file_utils.get_dirs_in_path(_path)
        len_folders = len(_dirs)

        if len_folders == 0:
            args["exp_folders"] = [args["exp_folder"]]
        else:
            # Sort out dirs not named *_perturbations
            args["exp_folders"] = [
                str(pl.Path(_dirs[i].parent.name, _dirs[i].name))
                for i in range(len_folders)
                if "rd" in _dirs[i].name
                or "nm" in _dirs[i].name
                or "rf" in _dirs[i].name
                # if "bv" in _dirs[i].name
                # if "perturbations" in _dirs[i].name  # or "nm" in _dirs[i].name
            ]

        # Update number of folders after filtering
        len_folders = len(args["exp_folders"])

    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    line_counter = 0
    perturb_type_old = ""
    for i, folder in enumerate(args["exp_folders"]):
        folder_path = pl.Path(folder)
        # Set exp_folder
        args["exp_folder"] = folder

        digits_in_name = lib_type_utils.get_digits_from_string(folder_path.name)
        if digits_in_name is not None:
            if isinstance(digits_in_name, int):
                perturb_type = folder_path.name.split(str(digits_in_name))[0]

                if not perturb_type == perturb_type_old:
                    color = cmap_list[i]
                    _save_color = color
                    perturb_type_old = perturb_type
                else:
                    color = _save_color
                    if digits_in_name >= args["n_runs_per_profile"]:
                        continue

                linestyle = LINESTYLES[digits_in_name]

        else:
            color = cmap_list[i]
            linestyle = None

        g_plt_data.plot_error_norm_vs_time(
            args,
            axes=axes[0],
            cmap_list=[color],
            linestyle=linestyle,
            legend_on=False,
            normalize_start_time=False,
            plot_args=[],
        )
        lines: list = list(axes[0].get_lines())
        lines[line_counter].set_label(str(pl.Path(folder).name))

        len_lines = len(lines)
        line_counter += len_lines - line_counter

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
        sh_plot.plot_energy(
            args,
            axes=axes[1],
            plot_args=[],
        )
    elif cfg.MODEL == Models.LORENTZ63:
        l63_plot.plot_energy(args, axes=axes[1])


def plot_exp_growth_rate_comparison(args: dict):
    """Plots a comparison of the exponential growth rates vs time for the different
    perturbation methods

    Parameters
    ----------
    args : dict
        Run-time arguments
    """

    # args["endpoint"] = True

    if args["exp_folders"] is not None:
        len_folders = len(args["exp_folders"])
    elif args["exp_folder"] is not None:
        # Get dirs in path
        _path = pl.Path(args["datapath"], args["exp_folder"])
        _dirs = lib_file_utils.get_dirs_in_path(_path)
        len_folders = len(_dirs)

        if len_folders == 0:
            args["exp_folders"] = [args["exp_folder"]]
        else:
            # Sort out dirs not named *_perturbations
            args["exp_folders"] = [
                str(pl.Path(_dirs[i].parent.name, _dirs[i].name))
                for i in range(len_folders)
                # if "sv" in _dirs[i].name
                if "perturbations" in _dirs[i].name  # or "nm" in _dirs[i].name
            ]

        # Update number of folders after filtering
        len_folders = len(args["exp_folders"])

    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=len_folders)
    # cmap_list[0] = "k"
    axes = plt.axes()

    perturb_type_old = ""
    for i, folder in enumerate(args["exp_folders"]):
        folder_path = pl.Path(folder)

        digits_in_name = lib_type_utils.get_digits_from_string(folder_path.name)
        if digits_in_name is not None:
            if isinstance(digits_in_name, int):

                perturb_type = folder_path.name.split(
                    lib_type_utils.zpad_string(str(digits_in_name), n_zeros=2)
                )[0]

                if not perturb_type == perturb_type_old:
                    color = cmap_list[i]
                    _save_color = color
                    perturb_type_old = perturb_type
                else:
                    color = _save_color
                    if digits_in_name >= args["n_runs_per_profile"]:
                        continue

                linestyle = LINESTYLES[digits_in_name]

        else:
            color = cmap_list[i]
            linestyle = None

        # Set exp_folder
        args["exp_folder"] = folder

        # if i == 0:
        #     zorder = 10
        # else:
        #     zorder = 0
        g_plt_data.plot_exp_growth_rate_vs_time(
            args=args,
            axes=axes,
            color=color,  # cmap_list[i],
            # zorder=zorder,
            linestyle=linestyle,
            anal_type="mean",
            plot_args=[],
            title_suffix=str(folder_path.parent),
        )

    if cfg.MODEL == Models.LORENTZ63:
        lower_bound: float = -16
        upper_bound: float = 4
        spacing: float = 2
        axes.set_yticks(
            np.linspace(
                lower_bound,
                upper_bound,
                int(abs(upper_bound - lower_bound) / spacing) + 1,
                endpoint=True,
            ),
            minor=False,
        )


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args: dict = compare_plot_arg_parser.args

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

    g_ui.confirm_run_setup(args)

    if "vec_compare" in args["plot_type"]:
        plt_vector_comparison(args)
    elif "error_norm_compare" in args["plot_type"]:
        plot_error_norm_comparison(args)
    elif "exp_growth_rate_compare" in args["plot_type"]:
        plot_exp_growth_rate_comparison(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args, tight_layout_rect=[0, 0, 0.9, 1])
