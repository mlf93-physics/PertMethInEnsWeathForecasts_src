import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from general.utils.module_import.type_import import *
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.analyses.analyse_data as g_a_data
import general.utils.util_funcs as g_utils
import general.utils.importing.import_perturbation_data as pt_import
from general.plotting.plot_params import *
from matplotlib.colors import LogNorm
from general.params.experiment_licences import Experiments as EXP
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    from shell_model_experiments.params.params import ParamsStructType
    from shell_model_experiments.params.params import PAR as PAR_SH
    import shell_model_experiments.utils.special_params as sh_sparams

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    params = l63_params
    sparams = l63_sparams


def plot_exp_growth_rate_vs_time(
    args=None,
    axes=None,
    exp_setup=None,
    linestyle: str = "-",
    linewidth: float = 2,
    alpha: float = 1.0,
    zorder: float = 0.0,
    color=None,
    legend_on: bool = True,
    title_suffix: str = "",
    anal_type: str = "instant",
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

    (
        mean_growth_rate,
        profile_mean_growth_rates,
    ) = g_a_data.execute_mean_exp_growth_rate_vs_time_analysis(
        args, u_stores, header_dicts=header_dicts, anal_type=anal_type
    )

    # Define time array
    # -1 since growth rate is a rate between differences (see functino
    # analyse_mean_exp_growth_rate_vs_time)
    time_array = np.linspace(
        0,
        header_dicts[0]["time_to_run"],
        int(header_dicts[0]["time_to_run"] * params.tts) + args["endpoint"] * 1 - 1,
        dtype=np.float64,
        endpoint=args["endpoint"],
    )

    # Prepare axes
    if axes is None:
        axes = plt.axes()

    if "detailed_label" in plot_args:
        label = args["exp_folder"]
    else:
        label = str(pl.Path(args["exp_folder"]).name).split("_perturbations")[0]
        if anal_type == "mean":
            label += f"; $\\lambda_{{mean}}$={mean_growth_rate[-1]:.2f}"

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
        title_header=f"{anal_type.capitalize()} " + "exponential growth rate vs time",
        title_suffix=title_suffix,
        detailed="detailed_title" in plot_args,
    )
    axes.set_title(title)
    axes.set_xlabel("$t$")
    axes.set_ylabel(
        "$\\kappa(t)$" if anal_type.lower() == "instant" else "$\\kappa_{{mean}}(t_0)$"
    )


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
    raw_perturbations: bool = True,
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
    ) = g_import.import_perturbation_velocities(
        args, search_pattern="*perturb*.csv", raw_perturbations=raw_perturbations
    )

    # Get number of runs per profile and n_profiles
    n_runs_per_profile = int(header_dicts[0]["n_runs_per_profile"])
    n_profiles = int(header_dicts[0]["n_profiles"])
    n_perturbations = len(perturb_time_pos_list)

    (
        error_norm_vs_time,
        error_norm_mean_vs_time,
    ) = g_a_data.analyse_error_norm_vs_time(u_stores, args=args)

    # print(
    #     "error_norm_vs_time",
    #     (
    #         error_norm_vs_time[-1, :]
    #         / error_norm_vs_time[0, :]
    #         / header_dicts[0]["time_to_run"]
    #     ).reshape((-1, args["n_runs_per_profile"])),
    # )

    # reshaped_error_norm_vs_time = np.reshape(error_norm_vs_time, (-1, 6, 20))
    # ens_mean_vs_time = np.mean(reshaped_error_norm_vs_time, axis=2)

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
        cmap_list, _ = g_plt_utils.get_non_repeating_colors(n_colors=n_colors)
    axes.set_prop_cycle("color", cmap_list)

    if header_dicts[0]["pert_mode"] in ["rd", "nm", "rf"]:
        linewidth: float = 1.0
        alpha: float = 0.5
        zorder: float = 5
    else:
        zorder: float = 10

    if header_dicts[0]["pert_mode"] == "nm":
        zorder = 7

    lines = axes.plot(
        time_array,
        error_norm_vs_time,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
    # axes.plot(time_array[:, np.s_[0:121:20]], ens_mean_vs_time, zorder=20, color="k")

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

    if cfg.MODEL == cfg.Models.LORENTZ63:
        ylabel_suffix = "$||\\mathbf{x} - \\mathbf{x}_{ref}||$"
    elif cfg.MODEL == cfg.Models.SHELL_MODEL:
        ylabel_suffix = "$||\\mathbf{u} - \\mathbf{u}_{ref}||$"

    axes.set_xlabel("Time")
    axes.set_ylabel("Error, " + ylabel_suffix)
    axes.set_yscale("log")

    if legend_on:
        if cfg.LICENCE not in [
            EXP.BREEDING_VECTORS,
            EXP.LYAPUNOV_VECTORS,
            EXP.SINGULAR_VECTORS,
        ]:
            axes.legend(perturb_time_pos_list_legend)
        elif cfg.LICENCE == EXP.SINGULAR_VECTORS:
            for i, header_dict in enumerate(header_dicts):
                if "run_in_profile" in header_dict:
                    lines[i].set_label(f"sv{int(header_dict['run_in_profile'])}")
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


def plot_RMSE_and_spread(
    data_array: np.ndarray,
    args: dict = None,
    header_dict: List[dict] = None,
    axes: plt.Axes = None,
    exp_setup=None,
    linestyle: str = "-",
    linewidth: float = 2,
    zorder: float = 0.0,
    color=None,
    legend_on: bool = True,
    label: str = "",
    detailed_title: bool = False,
):

    rmse_vs_time, spread_vs_time = g_a_data.analyse_RMSE_and_spread_vs_time(
        data_array, args=args
    )

    time_array = np.linspace(
        0,
        header_dict["time_to_run"],
        int(header_dict["time_to_run"] * params.tts) + args["endpoint"] * 1,
        dtype=np.float64,
        endpoint=args["endpoint"],
    )

    rmse_plot = axes.plot(
        time_array,
        rmse_vs_time,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
        label=label,
        color=color,
    )

    if args["rmse_spread"]:
        axes.plot(
            time_array,
            spread_vs_time,
            linestyle="dashed",
            zorder=zorder,
            color=rmse_plot[0].get_color(),
        )

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Average RMSE and ensemble spread vs time",
        detailed=detailed_title,
    )

    axes.set_xlabel("Time")
    axes.set_ylabel("RMSE and ensemble spread")
    axes.set_title(title)

    if legend_on:
        axes.legend()
    # axes.set_yscale("log")


def plot_energy(
    args,
    axes=None,
    zero_time_ref=None,
    plot_args=["detailed_title"],
    plot_kwargs={"exp_file_type": "perturbations"},
    linewidth=LINEWIDTHS["medium"],
    zorder=0,
):
    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    if axes is None:
        fig = plt.figure()
        axes = plt.axes()

    # Plot total energy vs time
    energy_vs_time = np.sum(u_data * np.conj(u_data), axis=1).real
    axes.plot(time.real, energy_vs_time, "k", zorder=zorder, linewidth=linewidth)
    axes.set_xlabel("Time")
    axes.set_ylabel("Energy, $\\frac{1}{2} u_{n, ref} u_{n, ref}^*$")
    # axes.set_xlim(args["ref_start_time"], args["ref_end_time"])

    header_dict = g_utils.handle_different_headers(header_dict)

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Energy vs. time",
        detailed="detailed_title" in plot_args,
    )
    axes.set_title(title)

    if "exp_folder" in args:
        if args["exp_folder"] is not None and args["mark_pert_start"]:
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


def plot2D_average_vectors(
    args, axes: plt.Axes = None, rel_mean_vector: bool = False, plot_kwargs: dict = {}
):
    (
        vector_units,
        characteristic_values,
        _,
        _,
        header_dicts,
    ) = pt_import.import_perturb_vectors(
        args, raw_perturbations=True, dtype=np.complex128
    )
    # print("vector_units", vector_units.shape)
    # print("np.abs(characteristic_values)", np.abs(characteristic_values))
    characteristic_values = (
        2 * 1 / header_dicts[0]["time_to_run"] * np.log(np.abs(characteristic_values))
    )
    sort_index = np.argsort(characteristic_values, axis=1)[:, ::-1]
    # print("sort_index", sort_index)
    for i in range(vector_units.shape[0]):
        vector_units[i, :, :] = vector_units[i, sort_index[i, :], :]
        characteristic_values[i, :] = characteristic_values[i, sort_index[i, :]]

    mean_lyapunov_exp = np.mean(characteristic_values, axis=0)
    print("mean_lyapunov_exp", mean_lyapunov_exp)
    plt.figure()
    # plt.plot(np.cumsum(mean_lyapunov_exp[1:]) / np.sum(mean_lyapunov_exp[1:]), ".")
    plt.plot(mean_lyapunov_exp, ".")  # / np.max(mean_lyapunov_exp[1:]), ".")
    # plt.title("Cummulative mean lyapunov exponents (normalized)")
    plt.title("Mean lyapunov exponents")
    plt.ylabel("$\\mu_m$")
    plt.xlabel("Lyapunov index, m")
    plt.figure()

    # Normalize
    vector_units = g_utils.normalize_array(vector_units, norm_value=1, axis=2)

    # Prepare axes
    if axes is None:
        axes = plt.axes()

    mean_abs_vector_units = np.mean(np.abs(vector_units), axis=0)

    if rel_mean_vector:
        mean_abs_vector_units = (
            mean_abs_vector_units
            - np.mean(mean_abs_vector_units, axis=0)[np.newaxis, :]
        )

    plot2D_vectors(mean_abs_vector_units, args, header_dicts, axes=axes, **plot_kwargs)


def plot2D_vectors(
    vector_units: np.ndarray,
    args: dict,
    header_dicts: List[dict],
    axes: plt.Axes = None,
    annot: bool = False,
    vmin: float = None,
    vmax: float = None,
    cmap="Reds",
    log_cmap: bool = False,
    xlabel: str = "x index",
    ylabel: str = "y index",
    title_header: str = "DEFAULT TITLE",
):

    # Prepare axes
    if axes is None:
        axes = plt.axes()

    sb.heatmap(
        vector_units.T,
        ax=axes,
        cmap=cmap,
        cbar_kws={
            "pad": 0.1,
        },
        annot=annot,
        fmt=".1f",
        annot_kws={"fontsize": 8},
        vmin=vmin,
        vmax=vmax,
        norm=LogNorm() if log_cmap else None,
    )

    axes.invert_yaxis()
    axes.invert_xaxis()
    plt.xticks(rotation=0)
    axes.yaxis.tick_right()
    axes.yaxis.set_label_position("right")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    # Generate title
    vector_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header=title_header,
    )
    axes.set_title(vector_title)
