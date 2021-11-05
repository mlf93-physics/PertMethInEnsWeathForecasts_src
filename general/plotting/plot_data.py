import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
from general.utils.module_import.type_import import *
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.utils.util_funcs as g_utils
from general.params.experiment_licences import Experiments as EXP
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = sh_params
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params


def analyse_error_norm_vs_time(u_stores, args=None):

    if len(u_stores) == 0:
        raise IndexError("Not enough u_store arrays to compare.")

    if args["combinations"]:
        combinations = [
            [j, i] for j in range(len(u_stores)) for i in range(j + 1) if j != i
        ]
        error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(combinations)))

        for enum, indices in enumerate(combinations):
            error_norm_vs_time[:, enum] = np.linalg.norm(
                u_stores[indices[0]] - u_stores[indices[1]], axis=1
            ).real
    else:
        error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(u_stores)))

        for i in range(len(u_stores)):
            if len(u_stores[i]) == 1:
                u_stores[i] = np.reshape(u_stores[i], (u_stores[i].size, 1))

            error_norm_vs_time[:, i] = np.linalg.norm(u_stores[i], axis=1).real

    error_norm_mean_vs_time = np.mean(error_norm_vs_time, axis=1)

    return error_norm_vs_time, error_norm_mean_vs_time


def analyse_error_spread_vs_time_norm_of_mean(u_stores, args=None):
    """Calculates the spread of the error (u' - u) using the 'norm of the mean.'

    Formula: ||\sqrt{<((u' - u) - <u' - u>)²>}||

    """

    u_mean = np.mean(np.array(u_stores), axis=0)

    error_spread = np.array([(u_stores[i] - u_mean) ** 2 for i in range(len(u_stores))])

    error_spread = np.sqrt(np.mean(error_spread, axis=0))
    error_spread = np.linalg.norm(error_spread, axis=1)

    return error_spread


def analyse_error_spread_vs_time_mean_of_norm(u_stores, args=None):
    """Calculates the spread of the error (u' - u) using the 'mean of the norm'

    Formula: \sqrt{<(||u' - u|| - <||u' - u||>)²>}

    """
    u_mean_norm = np.mean(np.linalg.norm(np.array(u_stores), axis=2).real, axis=0)

    error_spread = np.array(
        [
            (np.linalg.norm(u_stores[i], axis=1).real - u_mean_norm) ** 2
            for i in range(len(u_stores))
        ]
    )

    error_spread = np.sqrt(np.mean(error_spread, axis=0))

    return error_spread


def plot_error_norm_vs_time(
    args=None,
    normalize_start_time=True,
    axes=None,
    exp_setup=None,
    linestyle: str = "-",
    linewidth: float = 2,
    alpha: float = 1.0,
    zorder: float = 0.0,
    cmap_list: Union[None, list] = None,
    legend_on: bool = True,
    plot_args: list = ["detailed_title"],
):

    if exp_setup is None:
        try:
            exp_setup = g_import.import_exp_info_file(args)
        except ImportError:
            print(
                "\nThe .json config file was not found, so this plot doesnt work "
                + "if the file is needed\n"
            )

    (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        header_dicts,
        u_ref_stores,
    ) = g_import.import_perturbation_velocities(args, search_pattern="*perturb*.csv")

    num_perturbations = len(perturb_time_pos_list)

    error_norm_vs_time, error_norm_mean_vs_time = analyse_error_norm_vs_time(
        u_stores, args=args
    )
    if args["plot_mode"] == "detailed":
        error_spread_vs_time = analyse_error_spread_vs_time_mean_of_norm(
            u_stores, args=args
        )

    time_array = np.linspace(
        0,
        header_dicts[0]["time_to_run"],
        int(header_dicts[0]["time_to_run"] * params.tts) + args["endpoint"] * 1,
        dtype=np.float64,
        endpoint=args["endpoint"],
    )
    if not normalize_start_time:
        time_array = np.repeat(
            np.reshape(time_array, (time_array.size, 1)), num_perturbations, axis=1
        )

        time_array += np.reshape(
            np.array(perturb_time_pos_list) * params.stt, (1, num_perturbations)
        )

    # Pick out specified runs
    if args["specific_files"] is not None:
        perturb_time_pos_list_legend = [
            perturb_time_pos_list_legend[i] for i in args["specific_files"]
        ]
        error_norm_vs_time = error_norm_vs_time[:, args["specific_files"]]

    if args["plot_mode"] == "detailed":
        perturb_time_pos_list_legend = np.append(
            perturb_time_pos_list_legend, ["Mean error norm", "Std of error"]
        )

    # Prepare axes
    if axes is None:
        axes = plt.gca()

    # Get non-repeating colorcycle
    if cfg.LICENCE == EXP.BREEDING_VECTORS or cfg.LICENCE == EXP.LYAPUNOV_VECTORS:
        n_colors = exp_setup["n_vectors"]
    else:
        n_colors = num_perturbations

    # Set colors
    if cmap_list is None:
        cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=n_colors)
    axes.set_prop_cycle("color", cmap_list)

    if header_dicts[0]["pert_mode"] in ["rd", "nm"]:
        linewidth: float = 1.0
        alpha: float = 0.5
        zorder: float = 0
    else:
        zorder: float = 10

    axes.plot(
        time_array,
        error_norm_vs_time,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )

    if args["plot_mode"] == "detailed":
        # Plot perturbation error norms
        axes.plot(time_array, error_norm_mean_vs_time, "k")  # , 'k', linewidth=1)
        # Plot mean perturbation error norm
        axes.plot(time_array, error_spread_vs_time, "k--")  # , 'k', linewidth=1)
        # Plot std of perturbation errors

    axes.set_xlabel("Time")
    axes.set_ylabel("Error")
    axes.set_yscale("log")

    if legend_on:
        if cfg.LICENCE not in [EXP.BREEDING_VECTORS, EXP.LYAPUNOV_VECTORS]:
            axes.legend(perturb_time_pos_list_legend)

    if args["xlim"] is not None:
        axes.set_xlim(args["xlim"][0], args["xlim"][1])
    if args["ylim"] is not None:
        axes.set_ylim(args["ylim"][0], args["ylim"][1])

    title_suffix = ""
    if "shell_cutoff" in args:
        if args["shell_cutoff"] is not None:
            title_suffix = f" cutoff={args['shell_cutoff']}"

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Error vs time",
        title_suffix=title_suffix,
        detailed="detailed_title" in plot_args,
    )
    axes.set_title(title)

    return axes


def plot_energy(
    time=None,
    u_data=None,
    header_dict=None,
    axes=None,
    args=None,
    zero_time_ref=None,
    plot_args=["detailed_title"],
):
    # If data is not present, import it
    if time is None or u_data is None or header_dict is None:
        # Import reference data
        time, u_data, header_dict = g_import.import_ref_data(args=args)

    if axes is None:
        fig = plt.figure()
        axes = plt.axes()

    # Plot total energy vs time
    energy_vs_time = np.sum(u_data * np.conj(u_data), axis=1).real
    axes.plot(time.real, energy_vs_time, "k")
    axes.set_xlabel("Time")
    axes.set_ylabel("Energy")

    header_dict = g_utils.handle_different_headers(header_dict)

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Energy vs. time",
        detailed="detailed_title" in plot_args,
    )
    axes.set_title(title)

    if "exp_folder" in args:
        if args["exp_folder"] is not None:
            perturb_file_names = list(
                pl.Path(args["datapath"], args["exp_folder"]).glob("*.csv")
            )

            # Import headers to get perturb positions
            index = []
            for ifile, file_name in enumerate(perturb_file_names):
                header_dict = g_import.import_header(file_name=file_name)

                if zero_time_ref:
                    index.append(header_dict["perturb_pos"] - zero_time_ref)
                else:
                    index.append(header_dict["perturb_pos"])

                if ifile + 1 >= args["n_files"]:
                    break

            for idx in sorted(index):
                plt.plot(
                    idx * params.stt,
                    energy_vs_time[int(idx * params.sample_rate)],
                    marker="o",
                )

    return axes
