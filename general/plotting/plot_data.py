import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
from general.utils.module_import.type_import import *
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.analyses.analyse_data as g_a_data
import general.utils.util_funcs as g_utils
from general.plotting.plot_params import *
from general.params.experiment_licences import Experiments as EXP
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    from shell_model_experiments.params.params import ParamsStructType
    from shell_model_experiments.params.params import PAR as PAR_SH

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params

    params = l63_params


def plot_exp_growth_rate_vs_time(
    args=None,
    normalize_start_time=True,
    axes=None,
    exp_setup=None,
    linestyle: str = "-",
    linewidth: float = 2,
    alpha: float = 1.0,
    zorder: float = 0.0,
    color=None,
    legend_on: bool = True,
    title_suffix: str = "",
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

    # Define time array
    time_array = np.linspace(
        0,
        header_dicts[0]["time_to_run"],
        int(header_dicts[0]["time_to_run"] * params.tts) + args["endpoint"] * 1,
        dtype=np.float64,
        endpoint=args["endpoint"],
    )

    n_runs_per_profile = len(perturb_time_pos_list)

    # Analyse error norm and mean exponential growth rate
    (
        error_norm_vs_time,
        error_norm_mean_vs_time,
    ) = g_a_data.analyse_error_norm_vs_time(u_stores, args=args)

    mean_growth_rate = g_a_data.analyse_mean_exp_growth_rate_vs_time(
        error_norm_vs_time, args=args
    )

    # Prepare axes
    if axes is None:
        axes = plt.axes()

    if "detailed_label" in plot_args:
        label = args["exp_folder"]
    else:
        label = str(pl.Path(args["exp_folder"]).name)

    axes.plot(
        time_array,
        mean_growth_rate,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
        label=label,
        color=color,
    )

    if legend_on:
        axes.legend()

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Exponential growth rate vs time",
        title_suffix=title_suffix,
        detailed="detailed_title" in plot_args,
    )
    axes.set_title(title)
    axes.set_xlabel("Time")
    axes.set_ylabel("Exp. growth rate, $\\lambda$")


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

    # Get number of runs per profile and n_profiles
    n_runs_per_profile = int(header_dicts[0]["n_runs_per_profile"])
    n_profiles = int(header_dicts[0]["n_profiles"])
    n_perturbations = len(perturb_time_pos_list)

    (
        error_norm_vs_time,
        error_norm_mean_vs_time,
    ) = g_a_data.analyse_error_norm_vs_time(u_stores, args=args)
    if args["plot_mode"] == "detailed":
        error_spread_vs_time = g_a_data.analyse_error_spread_vs_time_mean_of_norm(
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
            np.reshape(time_array, (time_array.size, 1)), n_perturbations, axis=1
        )

        time_array += np.reshape(
            np.array(perturb_time_pos_list) * params.stt, (1, n_perturbations)
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
        axes = plt.axes()

    # Get non-repeating colorcycle
    if cfg.LICENCE in [
        EXP.LYAPUNOV_VECTORS,
        EXP.BREEDING_VECTORS,
    ]:
        n_colors = exp_setup["n_vectors"]
    else:
        n_colors = n_runs_per_profile
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

    if header_dicts[0]["pert_mode"] == "nm":
        zorder = 2

    lines = axes.plot(
        time_array,
        error_norm_vs_time,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )

    # Set unique linestyle
    if "unique_linestyle" in plot_args:
        for i, header_dict in enumerate(header_dicts):
            if "run_in_profile" in header_dict:
                lines[i].set_linestyle(LINESTYLES[int(header_dict["run_in_profile"])])
            else:
                raise KeyError(
                    "run_in_profile not found in header_dict. Unique linestyles cannot be made."
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
        if cfg.LICENCE not in [
            EXP.BREEDING_VECTORS,
            EXP.LYAPUNOV_VECTORS,
            EXP.SINGULAR_VECTORS,
        ]:
            axes.legend(perturb_time_pos_list_legend)
        elif cfg.LICENCE == EXP.SINGULAR_VECTORS:
            lines: list = list(axes.get_lines())
            for i, header_dict in enumerate(header_dicts):
                if "run_in_profile" in header_dict:
                    lines[i].set_label(f"sv{int(header_dict['run_in_profile'])}")
                    # lines[i].set_color(cmap_list[int(header_dict["run_in_profile"])])
                    if i + 1 == n_runs_per_profile:
                        break
            axes.legend()

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
    args,
    axes=None,
    zero_time_ref=None,
    plot_args=["detailed_title"],
    plot_kwargs={"exp_file_type": "perturbations"},
):
    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

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
            _, perturb_files = g_utils.get_exp_files_and_names(
                args, type=plot_kwargs["exp_file_type"]
            )

            # Import headers to get perturb positions
            index = []
            for ifile, file_name in enumerate(perturb_files):
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
