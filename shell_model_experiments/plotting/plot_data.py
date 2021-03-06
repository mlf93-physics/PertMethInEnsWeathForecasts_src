"""Make plots related to the standard shell model

Example
-------
python plotting/plot_data.py
--plot_type=error_norm
--exp_folder=temp1_test_no_hyper_diffusivity_ny_n16
--shell_cutoff=12
--endpoint
"""

import sys

sys.path.append("..")
from pathlib import Path
from typing import List
import pandas as pd
import config as cfg
import general.plotting.plot_data as g_plt_data
from general.utils.module_import.type_import import *
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_config as plt_config
from general.plotting.plot_params import *
import general.utils.user_interface as g_ui
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import matplotlib.colors as mpl_colors
import seaborn as sb
import numpy as np
import shell_model_experiments.utils.util_funcs as sh_utils
import shell_model_experiments.analyses.analyse_data as sh_analysis
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
from mpl_toolkits import mplot3d
from shell_model_experiments.params.params import ParamsStructType
from shell_model_experiments.params.params import PAR

# SHELL_TICKS_COMPACT = np.concatenate([np.array([1]), np.arange(2, PAR.sdim + 1, 2)])
SHELL_TICKS_COMPACT = np.arange(1, PAR.sdim, 2)
SHELL_TICKS_FULL = np.arange(1, PAR.sdim + 1, 1)


def plot_energy_spectrum(
    u_data: np.ndarray = None,
    header_dict: dict = None,
    axes: plt.Axes = None,
    plot_arg_list: list = ["kolmogorov", "calculate"],
    plot_kwargs: dict = {},
    args: dict = {},
):
    # Import reference data if not given as input
    if u_data is None:
        _, u_data, header_dict = g_import.import_ref_data(args=args)

    # Make axes if not present
    if axes is None:
        fig = plt.figure()
        axes = plt.axes()

    # React on plot_arg options
    if "calculate" in plot_arg_list:
        # Calculate mean energy
        mean_energy = sh_analysis.get_mean_energy(u_data)
    else:
        mean_energy = u_data.real.ravel()

    # Plot Kolmogorov scaling
    if "kolmogorov" in plot_arg_list:
        axes.plot(
            np.log2(PAR.k_vec_temp),
            1e2 * PAR.k_vec_temp ** (-5 / 3),
            "k--",
            label="$k^{{-5/3}}$",
        )

    label: str = (
        plot_kwargs["label"]
        if "label" in plot_kwargs
        else (
            f"$n_{{\\nu}}$={int(header_dict['ny_n'])}, "
            + f"$\\alpha$={int(header_dict['diff_exponent'])}"
        )
    )
    title_suffix: str = ""
    color = plot_kwargs["color"] if "color" in plot_kwargs else None
    linestyle = plot_kwargs["linestyle"] if "linestyle" in plot_kwargs else None

    # Fit the slope of the spectrum
    k_vectors = np.log2(PAR.k_vec_temp)
    if "fit_slope" in plot_arg_list or "rel_fit" in plot_arg_list:
        slope, intercept = sh_analysis.fit_spectrum_slope(mean_energy, header_dict)
    if "fit_slope" in plot_arg_list:
        label += f", slope={slope:.2e}, b={intercept:.2e}"
        slope_plot = axes.plot(k_vectors, np.exp(slope * k_vectors + intercept), "--")
        if color is None:
            color = slope_plot[0].get_color()

    # Normlize energy spectrum to fit
    if "rel_fit" in plot_arg_list:
        mean_energy /= np.exp(slope * np.log2(PAR.k_vec_temp) + intercept)
        title_suffix += "rel. fit, "

    mean_energy /= PAR.k_vec_temp

    # Plot energy spectrum
    axes.plot(k_vectors, mean_energy, label=label, color=color, linestyle=linestyle)

    # Axes setup
    axes.set_yscale("log")
    axes.set_xlabel("$n$")
    axes.set_ylabel("Spectral energy\ndensity, $E(n)$")
    axes.set_xlim(1, 20)
    axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    # Limit y axis if necessary
    min_mean_energy = np.min(mean_energy)
    if args["ylim"] is not None:
        axes.set_ylim(args["ylim"][0], args["ylim"][1])
    elif min_mean_energy > 0:
        if np.log(min_mean_energy) < -15:
            axes.set_ylim(1e-15, 10)
    else:
        axes.set_ylim(1e-12, 1e2)

    # Title setup
    if "title" in plot_kwargs:
        title = plot_kwargs["title"]
    else:
        title = g_plt_utils.generate_title(
            args,
            header_dict=header_dict,
            title_header="Energy spectrum",
            title_suffix=title_suffix,
        )
    axes.set_title(title)

    if args["tolatex"]:
        plt_config.adjust_axes(axes)

        axes.set_yticks([1e-10, 1e-6, 1e-2, 1e2])

        # if args["save_fig"]:
        # g_plt_utils.save_figure(
        #     subpath="thesis_figures/models/",
        #     file_name="sh_energy_spectrum",
        # )


def plot_helicity_spectrum(
    u_data: np.ndarray,
    header_dict: dict,
    args: dict,
    axes: plt.Axes = None,
    plot_arg_list: list = ["calculate", "hel_sign"],
    plot_kwargs: dict = {},
):
    # Make axes if not present
    if axes is None:
        fig = plt.figure()
        axes = plt.axes()

    if "calculate" in plot_arg_list:
        mean_helicity = sh_analysis.get_mean_helicity(u_data)
    else:
        mean_helicity = u_data.real.ravel()

    # Plot Kolmogorov scaling
    if "kolmogorov" in plot_arg_list:
        axes.plot(
            np.log2(PAR.k_vec_temp),
            PAR.k_vec_temp ** (1 / 3),
            "k--",
            label="$k^{{1/3}}$",
        )

    label: str = (
        plot_kwargs["label"]
        if "label" in plot_kwargs
        else (
            f"$n_{{\\nu}}$={int(header_dict['ny_n'])}, "
            + f"$\\alpha$={int(header_dict['diff_exponent'])}"
        )
    )
    title_suffix: str = ""
    color = plot_kwargs["color"] if "color" in plot_kwargs else None

    helicity_sign = np.array([i % 2 for i in range(PAR.sdim)], dtype=np.int)
    mean_helicity = np.abs(mean_helicity)

    # Fit the slope of the spectrum
    k_vectors = np.log2(PAR.k_vec_temp)
    if "fit_slope" in plot_arg_list or "rel_fit" in plot_arg_list:
        slope, intercept = sh_analysis.fit_spectrum_slope(mean_helicity, header_dict)
    if "fit_slope" in plot_arg_list:
        label += f", slope={slope:.2e}, b={intercept:.2e}"
        slope_plot = axes.plot(k_vectors, np.exp(slope * k_vectors + intercept), "--")
        if color is None:
            color = slope_plot[0].get_color()

    # Normlize energy spectrum to fit
    if "rel_fit" in plot_arg_list:
        mean_helicity /= np.exp(slope * np.log2(PAR.k_vec_temp) + intercept)
        title_suffix += "rel. fit, "

    # Plot curves
    axes.plot(
        np.log2(PAR.k_vec_temp),
        mean_helicity,
        label=label,
        color=color,
    )
    if "hel_sign" in plot_arg_list:
        # Plot annotation according to sign of helicity
        _offset = 1e4
        axes.scatter(
            np.log2(PAR.k_vec_temp)[helicity_sign == 1],
            _offset * np.ones(np.sum(helicity_sign == 1)),
            label="_nolegend_",
            marker="+",
            color="k",
        )
        axes.scatter(
            np.log2(PAR.k_vec_temp)[helicity_sign == 0],
            _offset * np.ones(np.sum(helicity_sign == 0)),
            label="_nolegend_",
            marker="_",
            color="k",
        )

    # Limit y axis if necessary
    min_mean_helicity = np.min(mean_helicity)
    if args["ylim"] is not None:
        axes.set_ylim(args["ylim"][0], args["ylim"][1])
    elif min_mean_helicity > 0:
        if np.log(min_mean_helicity) < -3:
            axes.set_ylim(1e-3, 1e5)
    else:
        axes.set_ylim(1e-3, 1e5)

    # Axes setup
    axes.set_yscale("log")
    axes.set_xlabel("k")
    axes.set_ylabel("Helicity")
    axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    # Title setup
    if "title" in plot_kwargs:
        title = plot_kwargs["title"]
    else:
        title = g_plt_utils.generate_title(
            args,
            header_dict=header_dict,
            title_header="Helicity spectrum",
            title_suffix=title_suffix,
        )
    axes.set_title(title)


def plot_velocity_spectrum(
    u_data: np.ndarray,
    header_dict: dict,
    axes: plt.Axes = None,
    plot_arg_list: list = ["kolmogorov", "calculate"],
    plot_kwargs: dict = {},
    args: dict = {},
):
    # Make axes if not present
    if axes is None:
        fig = plt.figure()
        axes = plt.axes()

    # React on plot_arg options
    if "calculate" in plot_arg_list:
        # Calculate mean velocity
        if "triad_mean" in plot_arg_list:
            # Triad mean
            velocity = np.mean(
                np.abs((u_data[:, :-2] * u_data[:, 1:-1] * u_data[:, 2:])) ** (1 / 3),
                axis=0,
            )
        else:
            # Shell mean
            velocity = np.mean(
                np.abs(u_data),
                axis=0,
            )
    else:
        velocity = u_data.real.ravel()

    # Plot Kolmogorov scaling
    if "kolmogorov" in plot_arg_list:
        if "triad_mean" in plot_arg_list:
            axes.plot(
                np.log2(PAR.k_vec_temp),
                np.exp(-8) * PAR.k_vec_temp ** (-1 / 2),
                "k--",
                label="$k^{{-1/2}}$",
            )
        else:
            axes.plot(
                np.log2(PAR.k_vec_temp),
                np.exp(-11) * PAR.k_vec_temp ** (-2 / 3),
                "k--",
                label="$k^{{-2/3}}$",
            )
            axes.plot(
                np.log2(PAR.k_vec_temp),
                np.exp(3) * PAR.k_vec_temp ** (-1 / 3),
                "k:",
                label="$k^{{-1/3}}$",
            )

    label: str = (
        plot_kwargs["label"]
        if "label" in plot_kwargs
        else (
            f"$n_{{\\nu}}$={int(header_dict['ny_n'])}, "
            + f"$\\alpha$={int(header_dict['diff_exponent'])}"
        )
    )
    title_suffix: str = ""
    color = plot_kwargs["color"] if "color" in plot_kwargs else None

    # Fit the slope of the spectrum
    k_vectors = np.log2(PAR.k_vec_temp)
    if "fit_slope" in plot_arg_list or "rel_fit" in plot_arg_list:
        slope, intercept = sh_analysis.fit_spectrum_slope(velocity, header_dict)
    if "fit_slope" in plot_arg_list:
        label += f", slope={slope:.2e}, b={intercept:.2e}"
        slope_plot = axes.plot(k_vectors, np.exp(slope * k_vectors + intercept), "--")
        if color is None:
            color = slope_plot[0].get_color()

    # Normlize energy spectrum to fit
    if "rel_fit" in plot_arg_list:
        velocity /= np.exp(slope * np.log2(PAR.k_vec_temp) + intercept)
        title_suffix += "rel. fit, "

    # Plot energy spectrum
    axes.plot(
        k_vectors[1:-1] if "triad_mean" in plot_arg_list else k_vectors,
        velocity,
        label=label if "triad_mean" not in plot_arg_list else None,
        color=color,
    )

    # Prepare strings
    if "triad_mean" in plot_arg_list:
        ylabel = "$\\langle U_n \\rangle$= $\\langle (u_{{n-1}} u_n u_{{n+1}})^{{1/3}}\\rangle$"
        title_header = "Mean U spectrum, inf. hyper diff."
    else:
        ylabel = "$\\langle u_n \\rangle$"
        title_header = "Velocity spectrum"

    # Axes setup
    axes.set_yscale("log")
    axes.set_xlabel("k")
    axes.set_ylabel(ylabel)
    axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    # Limit y axis if necessary
    min_velocity = np.min(velocity)
    if args["ylim"] is not None:
        axes.set_ylim(args["ylim"][0], args["ylim"][1])
    elif min_velocity > 0:
        if np.log(min_velocity) < -6:
            axes.set_ylim(1e-6, 10)
    else:
        axes.set_ylim(1e-6, 10)

    # Title setup
    if "title" in plot_kwargs:
        title = plot_kwargs["title"]
    else:
        title = g_plt_utils.generate_title(
            args,
            header_dict=header_dict,
            title_header=title_header,
            title_suffix=title_suffix,
        )
    axes.set_title(title)
    plt.legend()


def plot_energy_per_shell(ax=None, omit=None, path=None, args=None):
    if ax is None:
        ax = plt.axes()

    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    if isinstance(args["shell_cutoff"], int):
        u_data = u_data[:, : args["shell_cutoff"]]

    # Plot total energy vs time
    energy_vs_time = np.cumsum((u_data * np.conj(u_data)).real, axis=1)
    ax.plot(time.real, energy_vs_time, "k")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Cummulative shell energy vs time",
    )

    ax.set_title(title)

    if args["exp_folder"] is not None:
        # file_names = list(Path(path).glob('*.csv'))
        # Find reference file
        # ref_file_index = None
        # for ifile, file in enumerate(file_names):
        #     file_name = file.stem
        #     if file_name.find('ref') >= 0:
        #         ref_file_index = ifile

        # if ref_file_index is None:
        #     raise ValueError('No reference file found in specified directory')

        perturb_file_names = list(
            Path(args["datapath"], args["exp_folder"]).glob("*.csv")
        )

        (
            pert_u_stores,
            perturb_time_pos_list,
            perturb_time_pos_list_legend,
            header_dicts,
            _,
        ) = g_import.import_perturbation_velocities(args)

        index = []
        for header_dict in header_dicts:
            index.append(header_dicts[-1]["perturb_pos"])

        header_dicts = [header_dicts[i] for _, i in enumerate(np.argsort(index))]
        index = [index[i] for _, i in enumerate(np.argsort(index))]

        for idx in range(len(index)):

            point_plot = plt.plot(
                np.ones(PAR.sdim) * header_dicts[idx]["perturb_pos"] * PAR.stt,
                energy_vs_time[int(header_dicts[idx]["perturb_pos"])],
                "o",
            )

            time_array = np.linspace(
                0,
                header_dicts[idx]["time_to_run"],
                int(header_dicts[idx]["time_to_run"] * PAR.tts) + args["endpoint"] * 1,
                dtype=np.float64,
                endpoint=args["endpoint"],
            )

            perturbation_energy_vs_time = np.cumsum(
                (
                    (
                        pert_u_stores[idx]
                        + u_data[
                            int(header_dicts[idx]["perturb_pos"]) : int(
                                header_dicts[idx]["perturb_pos"]
                            )
                            + int(header_dicts[idx]["N_data"]),
                            :,
                        ]
                    )
                    * np.conj(
                        pert_u_stores[idx]
                        + u_data[
                            int(header_dicts[idx]["perturb_pos"]) : int(
                                header_dicts[idx]["perturb_pos"]
                            )
                            + int(header_dicts[idx]["N_data"]),
                            :,
                        ]
                    )
                ).real,
                axis=1,
            )
            ax.plot(
                time_array + perturb_time_pos_list[idx] * PAR.stt,
                perturbation_energy_vs_time,
                color=point_plot[0].get_color(),
            )

            if idx + 1 >= args["n_files"]:
                break


def plot_energy(args=None, axes=None, plot_args=["detailed_title"], zorder=0):

    # plot_energy_spectrum(u_data, header_dict, args=args)
    axes = g_plt_data.plot_energy(
        args,
        axes=axes,
        plot_args=plot_args,
        zorder=zorder,
        linewidth=LINEWIDTHS["thin"],
    )

    if args["tolatex"]:
        plt_config.adjust_axes(axes)
        # plt_config.adjust_axes_to_subplot(axes)
        if cfg.MODEL == cfg.Models.LORENTZ63:
            plt_config.set_axis_tick_format(axes.yaxis)
        elif cfg.MODEL == cfg.Models.SHELL_MODEL:
            plt_config.set_axis_tick_format(axes.yaxis, fmt="%.1f")

        # if args["save_fig"]:
        #     g_plt_utils.save_figure(
        #         subpath="thesis_figures/models/",
        #         file_name="sh_energy_vs_time",
        #     )
    return axes


def plot_shell_error_vs_time(args=None):

    # Force max on files
    max_files = 3
    if args["n_files"] > max_files:
        args["n_files"] = max_files

    (
        u_stores,
        _,
        perturb_time_pos_list_legend,
        header_dicts,
        _,
    ) = g_import.import_perturbation_velocities(args)

    time_array = np.linspace(
        0,
        header_dicts[0]["time_to_run"],
        int(header_dicts[0]["time_to_run"] * PAR.tts),
        dtype=np.float64,
        endpoint=False,
    )

    for i in range(len(u_stores)):
        plt.figure()
        plt.plot(time_array, np.abs(u_stores[i]))  # , axis=1))
        plt.xlabel("Time [s]")
        plt.ylabel("Error")
        plt.yscale("log")
        plt.ylim(1e-16, 10)
        # plt.xlim(0.035, 0.070)
        plt.legend(PAR.k_vec_temp)

        title = g_plt_utils.generate_title(
            args,
            header_dict=header_dicts[0],
            title_header="Shell error vs time",
        )
        plt.title(title)


def plot_eigen_value_dist(args=None, axes=None, fig=None):

    if axes is None:
        fig = plt.figure()
        axes = plt.axes()

    u_init_profiles, perturb_positions, header_dict = g_import.import_start_u_profiles(
        args=args
    )

    _, e_vector_collection, e_value_collection = sh_nm_estimator.find_normal_modes(
        u_init_profiles,
        args,
        dev_plot_active=False,
        local_ny=header_dict["ny"],
    )

    # exit()
    perturb_time_pos_list = []
    # Sort eigenvalues
    for i in range(len(e_value_collection)):
        sort_id = e_value_collection[i].argsort()[::-1]
        e_value_collection[i] = e_value_collection[i][sort_id]

        # Prepare legend
        perturb_time_pos_list.append(
            f"Time: {perturb_positions[i]/PAR.sample_rate*PAR.dt:.1f}s"
        )

    e_value_collection = np.array(e_value_collection, dtype=np.complex128).T

    kolm_sinai_entropy = sh_utils.get_kolm_sinai_entropy(e_value_collection)

    # Plot normalised sum of eigenvalues
    krange = np.log2(PAR.k_vec_temp) - 0.5
    cumsummed_evalues = np.mean(
        np.cumsum(e_value_collection.real, axis=0) / kolm_sinai_entropy, axis=1
    )
    print("e_value_collection.real", np.mean(e_value_collection.real, axis=1))
    axes.plot(krange[:4], cumsummed_evalues[:4], "k+")
    axes.plot(krange[4:12], cumsummed_evalues[4:12], "k.")
    axes.plot(krange[12:], cumsummed_evalues[12:], "k_")
    # max_evalue = np.max(mean_evalues)
    # scatter_plot = axes.scatter(
    #     krange,
    #     cumsummed_evalues,
    #     c=mean_evalues,
    #     cmap=plt.cm.jet,
    #     vmax=max_evalue,
    #     vmin=mean_evalues[15],
    # )
    axes.set_xlabel("$m$")
    axes.set_ylabel("$\sum_{i=0}^m \\Re(\\mu_i) / H$")
    axes.set_ylim(-1, 1.1)
    axes.set_xlim(0, 20)
    # axes.legend(perturb_time_pos_list)
    axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    # fig.colorbar(
    #     scatter_plot,
    #     ax=axes,
    #     orientation="horizontal",
    #     pad=0.1,
    # )
    # fig = plt.gcf()
    # cbar_ax = fig.axes[-1]
    # cbar_ax.set_title("$\\Re(\\mu_m)$", x=-0.17, y=-2.0)

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Cummulative eigenvalues",
        title_suffix=f'N_tot={args["n_profiles"]*args["n_runs_per_profile"]}, ',
    )

    # axes.grid(visible=True, axis="y", color="k", linestyle="dotted")

    # plt.title(title)

    if args["tolatex"]:
        plt_config.adjust_axes_to_subplot(axes)


def plot_3D_eigen_mode_anal_comparison(args: dict = None):

    # Prepare axes
    fig1, axes1 = plt.subplots(
        ncols=len(args["datapaths"]),
        subplot_kw={"projection": "3d"},
        figsize=(16, 9),
    )
    fig2, axes2 = plt.subplots(ncols=len(args["datapaths"]), figsize=(16, 9))

    # Prepare plot settings
    mpl_ticker.MaxNLocator.default_params["integer"] = True

    for i, path in enumerate(args["datapaths"]):
        # Prepare datapaths
        args["datapath"] = path

        plot_3D_eigen_mode_analysis(args, axes=[axes1[i], axes2[i]], figs=[fig1, fig2])


def plot_3D_eigen_mode_analysis(
    args: dict = None,
    right_handed: bool = True,
    axes: List[plt.Axes] = None,
    figs: List[plt.Figure] = None,
    compare_plot: bool = False,
):
    if axes is None:
        fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
        fig2, ax2 = plt.figure(), plt.axes()
        axes = [ax1, ax2]
        figs = [fig1, fig2]
    else:
        if len(axes) != 2:
            raise ValueError("Not the appropriate number of axes")

    u_init_profiles, header_dict = pt_import.import_profiles_for_nm_analysis(args)

    # Set diff_exponent
    args["diff_exponent"] = header_dict["diff_exponent"]

    _, e_vector_collection, e_value_collection = sh_nm_estimator.find_normal_modes(
        u_init_profiles,
        args,
        dev_plot_active=False,
        local_ny=header_dict["ny"],
    )

    for i in range(len(e_value_collection)):
        sort_id = e_value_collection[i].argsort()[::-1]
        e_vector_collection[i] = e_vector_collection[i][:, sort_id]

    e_vector_collection = np.array(e_vector_collection)

    # Make data.
    shells = np.arange(0, PAR.sdim, 1)
    lyaponov_index = np.arange(0, PAR.sdim, 1)
    lyaponov_index, shells = np.meshgrid(lyaponov_index, shells)

    surf_plot = axes[0].plot_surface(
        lyaponov_index,
        shells,
        np.mean(np.abs(e_vector_collection) ** 2, axis=0),
        cmap="Reds",
    )
    axes[0].set_xlabel("Lyaponov index, $j$")
    axes[0].set_ylabel("Shell number, $i$")
    axes[0].set_zlabel("$<|v_i^j|^2>$")
    axes[0].set_zlim(0, 1)
    axes[0].view_init(elev=27.0, azim=-21)
    axes[0].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axes[0].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axes[0].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axes[0].xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
    axes[0].yaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
    axes[0].grid(False)
    figs[0].colorbar(surf_plot, ax=axes[0], pad=0.15, fraction=0.025)

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Eigenvectors vs shell numbers",
        title_suffix=f'N_tot={args["n_profiles"]*args["n_runs_per_profile"]}, ',
    )

    axes[0].set_title(title)

    # Set axis limits
    if right_handed:
        axes[0].set_xlim(PAR.sdim, 0)
    else:
        axes[0].set_xlim(0, PAR.sdim)
    axes[0].set_ylim(0, PAR.sdim)

    pcolorplot = axes[1].pcolormesh(
        np.mean(np.abs(e_vector_collection) ** 2, axis=0), cmap="Reds"
    )
    axes[1].set_xlabel("Lyaponov index")
    axes[1].set_ylabel("Shell number")
    axes[1].set_title(title)

    if right_handed:
        axes[1].set_xlim(PAR.sdim, 0)
        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")
        axes[1].xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
        axes[1].yaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
        figs[1].colorbar(pcolorplot, pad=0.1, ax=axes[1])
    else:
        figs[1].colorbar(pcolorplot, ax=axes[1])

    pcolorplot.set_clim(0, 1)


def plot_2D_eigen_mode_analysis(
    args: dict = None,
    right_handed: bool = False,
    axes: plt.Axes = None,
    fig: plt.Figure = None,
):
    if axes is None:
        fig, axes = plt.figure(), plt.axes()

    u_init_profiles, header_dict = pt_import.import_profiles_for_nm_analysis(args)

    # Set diff_exponent
    args["diff_exponent"] = header_dict["diff_exponent"]

    _, e_vector_collection, e_value_collection = sh_nm_estimator.find_normal_modes(
        u_init_profiles,
        args,
        dev_plot_active=False,
        local_ny=header_dict["ny"],
    )

    for i in range(len(e_value_collection)):
        sort_id = e_value_collection[i].argsort()[::-1]
        e_vector_collection[i] = e_vector_collection[i][:, sort_id]

    e_vector_collection = np.array(e_vector_collection)

    # Make data.
    # shells = np.arange(1, PAR.sdim + 1, 1)
    # lyaponov_index = np.arange(1, PAR.sdim + 1, 1)
    # lyaponov_index, shells = np.meshgrid(lyaponov_index, shells)

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Eigenvectors vs shell numbers",
        title_suffix=f'N_tot={args["n_profiles"]*args["n_runs_per_profile"]}, ',
    )

    mean_e_vectors = np.mean(np.abs(e_vector_collection), axis=0)
    eps = 2 * 1e-2
    pcolorplot = sb.heatmap(
        mean_e_vectors,
        cmap="Reds",
        mask=mean_e_vectors < eps,
        vmin=0,
        vmax=1,
        cbar=True,
        cbar_kws={
            "location": "bottom",
            "shrink": 0.5,
            "label": "$\\langle|\\xi_{n,m}| \\rangle$",
        },
    )
    pcolorplot.set_yticks(SHELL_TICKS_COMPACT)
    pcolorplot.set_yticklabels(SHELL_TICKS_COMPACT, rotation=0)
    pcolorplot.set_xticklabels(SHELL_TICKS_FULL)
    axes.set_xlabel("$m$")
    axes.set_ylabel("$n$")
    axes.set_title(title)
    axes.invert_yaxis()

    if right_handed:
        axes.set_xlim(PAR.sdim, 0)
        axes.yaxis.tick_right()
        axes.yaxis.set_label_position("right")

    # axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
    # axes.yaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
    # fig.colorbar(pcolorplot, pad=0.1, ax=axes, location="bottom", shrink=0.5)
    # else:
    # fig.colorbar(pcolorplot, ax=axes, location="bottom", shrink=0.5)

    if args["tolatex"]:
        plt_config.adjust_axes(axes)


def plot_eigen_vector_comparison(args=None):
    u_init_profiles, perturb_positions, header_dict = g_import.import_start_u_profiles(
        args=args
    )

    _, e_vector_collection, e_value_collection = sh_nm_estimator.find_normal_modes(
        u_init_profiles,
        args,
        dev_plot_active=False,
        local_ny=header_dict["ny"],
    )

    # Sort eigenvectors
    for i in range(len(e_value_collection)):
        sort_id = e_value_collection[i].argsort()[::-1]
        e_vector_collection[i] = e_vector_collection[i][:, sort_id]

    e_vector_collection = np.array(e_vector_collection, np.complex128)
    # current_e_vectors = np.mean(e_vector_collection, axis=0)
    # current_e_vectors = e_vector_collection[1]

    dev_plot = False

    integral_mean_lyaponov_index = (
        8  # int(np.average(np.arange(current_e_vectors.shape[1])
    )
    # , weights=np.abs(current_e_vectors[0, :])))

    print("integral_mean_lyaponov_index", integral_mean_lyaponov_index)

    orthogonality_array = np.zeros(
        (e_vector_collection.shape[0], integral_mean_lyaponov_index - 1),
        dtype=np.complex128,
    )

    for j in range(e_vector_collection.shape[0]):
        current_e_vectors = e_vector_collection[j]

        for i in range(1, integral_mean_lyaponov_index):
            # print(f'{integral_mean_lyaponov_index - i} x'+
            #     f' {integral_mean_lyaponov_index + i} : ',
            orthogonality_array[j, i - 1] = np.vdot(
                current_e_vectors[:, integral_mean_lyaponov_index - i],
                current_e_vectors[:, integral_mean_lyaponov_index + i],
            )

    # Scatter plot
    legend = []

    fig, axes = plt.subplots(1, 1)

    for i in range(integral_mean_lyaponov_index - 1):
        # plt.scatter(
        #     orthogonality_array[i : -1 : (integral_mean_lyaponov_index - 1)].real,
        #     orthogonality_array[i : -1 : (integral_mean_lyaponov_index - 1)].imag,
        #     marker=".",
        # )
        axes.hist(orthogonality_array[:, i].real, density=True, histtype="step")

        legend.append(f"$\\Delta$={i + 1}")

    axes.set_xlabel("$\\Re(\\langle \\xi_{9-\\Delta}; \\xi_{9 + \\Delta} \\rangle)$")
    axes.set_ylabel("Frequency")
    axes.legend(legend)

    if args["tolatex"]:
        # plt_config.remove_legends(axes)
        plt_config.adjust_axes(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            fig=fig,
            subpath="thesis_figures/models/",
            file_name="sh_orthogonality_of_phasespace",
        )


def plot_pert_traject_energy_spectrum(args):

    (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        header_dicts,
        _,
    ) = g_import.import_perturbation_velocities(args, raw_perturbations=False)

    axes = plt.axes()

    for u_data in u_stores:
        plot_energy_spectrum(u_data, header_dicts[0], axes=axes, args=args)


def plot_error_energy_spectrum_vs_time_2D(args: dict = None, axes: plt.Axes = None):

    if axes is None:
        axes = plt.axes()

    (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        header_dicts,
        _,
    ) = g_import.import_perturbation_velocities(args)

    if args["n_files"] == np.inf:
        n_files = len(perturb_time_pos_list)
    else:
        n_files = args["n_files"]

    n_divisions = 25

    # Prepare exponential time indices
    time_linear = np.linspace(0, 10, n_divisions)
    time_exp_indices = np.array(
        header_dicts[0]["N_data"] / np.exp(10) * np.exp(time_linear), dtype=np.int32
    )
    time_exp_indices[-1] -= 1  # Include endpoint manually

    # Store only unique indices
    time_exp_indices = np.unique(time_exp_indices)
    # Update numbe of divisions
    n_divisions = time_exp_indices.size
    error_spectra = np.zeros((n_files, n_divisions, PAR.sdim), dtype=np.float64)

    for ifile in range(n_files):
        for i, data_index in enumerate(time_exp_indices):
            error_spectra[ifile, i, :] = np.abs(u_stores[ifile][data_index, :]).real

    # Calculate mean and std
    error_mean_spectra = np.zeros((n_divisions, PAR.sdim), dtype=np.float64)
    # Find zeros
    error_spectra[np.where(error_spectra == 0)] = np.nan

    for i, data_index in enumerate(time_exp_indices):
        error_mean_spectra[i, :] = np.nanmean(error_spectra[:, i, :], axis=0)

    # error_mean_spectra[np.where(error_mean_spectra == np.nan)] = 0.0
    # error_mean_spectra[0, :] = error_spectra[0, 0, :]

    cmap_list, _ = g_plt_utils.get_non_repeating_colors(n_colors=n_divisions)
    axes.set_prop_cycle("color", cmap_list)
    axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    axes.plot(np.log2(PAR.k_vec_temp), error_mean_spectra.T)
    axes.plot(
        np.log2(PAR.k_vec_temp), PAR.k_vec_temp ** (-1 / 3), "k--", label="$k^{2/3}$"
    )
    axes.set_yscale("log")
    legend = [f"{item/PAR.sample_rate*PAR.dt:.3e}" for item in time_exp_indices]
    # axes.legend(legend, loc="center right", bbox_to_anchor=(1.3, 0.5))
    axes.set_xlabel("Shell number")
    axes.set_ylabel("$u_n - u^{'}_n$")
    axes.set_ylim(1e-22, 10)

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Error energy spectrum vs time",
        title_suffix=f"N_tot={n_files}, ",
    )
    axes.set_title(title)


def plot_error_vector_spectrogram(args=None):
    args["n_files"] = 1
    # args['file_offset'] = 0

    # Import perturbation data
    (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        _,
    ) = g_import.import_perturbation_velocities(args)

    args["start_times"] = np.array(
        [perturb_time_pos_list[args["file_offset"]] * PAR.stt],
        dtype=np.float64,
    )
    args["n_profiles"] = len(args["start_times"])

    # Import start u profiles at the perturbation
    u_init_profiles, perturb_positions, header_dict = g_import.import_start_u_profiles(
        args=args
    )

    _, e_vector_collection, e_value_collection = sh_nm_estimator.find_normal_modes(
        u_init_profiles,
        args,
        dev_plot_active=False,
        local_ny=header_dict["ny"],
    )

    sort_id = e_value_collection[0].argsort()[::-1]

    error_spectrum = (np.linalg.inv(e_vector_collection[0]) @ u_stores[0].T).real
    error_spectrum = error_spectrum / np.linalg.norm(error_spectrum, axis=0)

    # Make spectrogram
    plt.figure()
    # time = np.linspace(0, perturb_header_dicts[0]['N_data']*PAR.dt/PAR.sample_rate, 10)
    # x, y = np.meshgrid(time, np.log2(PAR.k_vec_temp))
    plt.pcolormesh(np.abs(error_spectrum[sort_id, :]), cmap="Reds")
    plt.xlabel("Time")
    # plt.xticks(time)
    plt.ylabel("Lyaponov index, j")
    plt.title(
        f'Error spectrum vs time; f={perturb_header_dicts[0]["f"]}'
        + f', $n_f$={int(perturb_header_dicts[0]["n_f"])}, $\\nu$={perturb_header_dicts[0]["ny"]:.2e}'
        + f', time={perturb_header_dicts[0]["time_to_run"]}s'
    )  # , N_tot={args["n_profiles"]*args["n_runs_per_profile"]}')
    plt.colorbar(label="$|c_j|/||c||$)")
    plt.savefig(
        f'../figures/week6/error_eigen_spectrogram/error_eigen_spectrogram_ny{header_dict["ny"]:.2e}_file_{args["file_offset"]}',
        format="png",
    )

    # plt.savefig(f'../figures/week6/error_eigen_value_spectra_2D/error_eigen_value_spectrum_ny{header_dict["ny"]}_time_{i/u_stores[0].shape[0]}.png', format='png')
    # plt.clim(0, 1)


def plot_error_vector_spectrum(args=None):
    args["n_files"] = 2
    # args['file_offset'] = 0

    # Import perturbation data
    (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        _,
    ) = g_import.import_perturbation_velocities(args)

    args["start_times"] = np.array(
        [
            perturb_time_pos_list[args["file_offset"] + i] * PAR.stt
            for i in range(args["n_files"])
        ],
        dtype=np.float64,
    )
    args["n_profiles"] = len(args["start_times"])

    print("start_times", args["start_times"])

    # Import start u profiles at the perturbation
    u_init_profiles, perturb_positions, header_dict = g_import.import_start_u_profiles(
        args=args
    )

    _, e_vector_collection, e_value_collection = sh_nm_estimator.find_normal_modes(
        u_init_profiles,
        args,
        dev_plot_active=False,
        local_ny=header_dict["ny"],
    )

    sorted_time_and_pert_mean_scaled_e_vectors = np.zeros(
        (args["n_files"], PAR.sdim, PAR.sdim)
    )

    for j in range(args["n_files"]):
        sort_id = e_value_collection[j].argsort()[::-1]

        error_spectrum = (np.linalg.inv(e_vector_collection[j]) @ u_stores[j].T).real
        error_spectrum = error_spectrum / np.linalg.norm(error_spectrum, axis=0)

        # Make average spectrum
        scaled_e_vectors = np.array(
            [
                e_vector_collection[j] * error_spectrum[:, i]
                for i in range(error_spectrum.shape[1])
            ]
        )
        scaled_e_vectors = np.abs(scaled_e_vectors) ** 2
        mean_scaled_e_vectors = np.mean(scaled_e_vectors, axis=0)
        sorted_mean_scaled_e_vectors = mean_scaled_e_vectors[:, sort_id]
        sorted_time_and_pert_mean_scaled_e_vectors[
            j, :, :
        ] = sorted_mean_scaled_e_vectors

    sorted_time_and_pert_mean_scaled_e_vectors = np.mean(
        sorted_time_and_pert_mean_scaled_e_vectors, axis=0
    )

    fig = plt.figure()
    axes = plt.axes()
    plt.pcolormesh(sorted_time_and_pert_mean_scaled_e_vectors, cmap="Reds")
    plt.xlabel("Lyaponov index")
    plt.ylabel("Shell number")
    plt.title(
        f'Eigenvectors vs shell numbers; f={header_dict["f"]}'
        + f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'
        + f', time={header_dict["time_to_run"]}s, N_tot={args["n_profiles"]*args["n_runs_per_profile"]}'
    )
    plt.xlim(PAR.sdim, 0)
    axes.yaxis.tick_right()
    axes.yaxis.set_label_position("right")
    plt.colorbar(pad=0.1)


def plot_howmoller_diagram_u_energy(
    args=None, axes=None, fig=None, plt_args: list = ["rel_mean"]
):

    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    # Prepare mesh
    time2D, shell2D = np.meshgrid(time.real, PAR.k_vec_temp)
    energy_array = (u_data * np.conj(u_data)).real.T

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1)

    if "rel_mean" in plt_args:
        # Prepare energy rel mean
        mean_energy = np.reshape(np.mean(energy_array, axis=1), (PAR.sdim, 1))
        energy_rel_mean_array = energy_array - mean_energy
        # energy_rel_mean_array = np.clip(energy_rel_mean_array, -15, None)

        # Get cmap
        cmap, norm = g_plt_utils.get_custom_cmap(
            vmin=np.min(energy_rel_mean_array),
            vmax=np.max(energy_rel_mean_array),
            neg_thres=0.4,
            pos_thres=0.6,
            cmap_handle=plt.cm.bwr,
        )
    else:
        cmap = "Reds"
        norm = None
        energy_rel_mean_array = np.log(energy_array)

    # Make contourplot
    pcm = axes.contourf(
        time2D,
        np.log2(shell2D),
        energy_rel_mean_array,
        norm=norm,
        cmap=cmap,
        levels=30,
    )
    pcm.negative_linestyle = "solid"
    axes.set_ylabel("$n$")
    axes.set_xlabel("Time [tu]")
    axes.set_yticks([1, 10, 20])
    axes.set_xlim(args["ref_start_time"], args["ref_end_time"])
    # axes.yaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    title_header = "Howm??ller diagram for $|u|^2$"
    title_header += "- $\\langle|u|^2\\rangle_t$" if "rel_mean" in plt_args else ""
    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header=title_header,
    )

    axes.set_title(title)

    # cbar_label = "$|u|??$"
    # cbar_label += " - $\\langle|u|??\\rangle_t$" if "rel_mean" in plt_args else ""
    cbar_label = "$|u_n|^2 - \\langle|u_n|^2\\rangle_t$"
    cbar = fig.colorbar(
        pcm,
        ax=axes,
        pad=0.3,
        label=cbar_label,
        shrink=0.5,
        location="bottom",
        ticks=[-3, -1.5, 0, 1.5, 3],
    )

    fig.subplots_adjust(
        top=0.961, bottom=0.053, left=0.15, right=0.967, hspace=0.614, wspace=0.2
    )

    if args["tolatex"]:
        plt_config.adjust_axes(axes)

        # if args["save_fig"]:
        #     g_plt_utils.save_figure(
        #         subpath="thesis_figures/models/",
        #         file_name="sh_howmoller_vs_time",
        #     )


def plot_howmoller_diagram_helicity(args=None, plt_args: list = ["rel_mean"]):

    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    # Prepare mesh
    time2D, shell2D = np.meshgrid(time.real, PAR.k_vec_temp)
    helicity_array = (PAR.hel_pre_factor * (u_data * np.conj(u_data)).real).T

    # Take absolute value
    helicity_array = np.abs(helicity_array)

    if "rel_mean" in plt_args:
        # Prepare helicity rel mean
        mean_helicity = np.reshape(np.mean(helicity_array, axis=1), (PAR.sdim, 1))
        mean_helicity = np.abs(mean_helicity)
        helicity_rel_mean_array = np.log(helicity_array) - np.log(mean_helicity)
    else:
        helicity_rel_mean_array = np.log(helicity_array)

    fig, axes = plt.subplots(nrows=1, ncols=1)

    if "rel_mean" in plt_args:
        vmax = np.nanmax(helicity_rel_mean_array)
        vmin = -6
        helicity_rel_mean_array = np.clip(helicity_rel_mean_array, vmin, vmax)

        # Get cmap
        cmap, norm = g_plt_utils.get_custom_cmap(
            vmin=vmin,
            vmax=vmax,
            neg_thres=0.4,
            pos_thres=0.6,
            cmap_handle=plt.cm.bwr,
        )
    else:
        cmap = "Reds"
        norm = None

    # Make contourplot
    pcm = axes.contourf(
        time2D,
        np.log2(shell2D),
        helicity_rel_mean_array,
        norm=norm,
        cmap=cmap,
        levels=30,
        extend="neither",
    )
    pcm.negative_linestyle = "solid"
    axes.set_ylabel("Shell number, n")
    axes.set_xlabel("Time")

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Howm??ller diagram for helicity",
        title_suffix="rel. mean helicity, " if "rel_mean" in plt_args else "",
    )

    axes.set_title(title)
    cbar_label = "log[H]"
    cbar_label += "- log[H_mean]" if "rel_mean" in plt_args else ""
    fig.colorbar(
        pcm,
        ax=axes,
        pad=0.1,
        label=cbar_label,
    )


def plot_eddie_turnover_times(args=None):
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    mean_eddy_turnover = sh_analysis.get_eddy_turnovertime(u_data)
    # eddy_freq = 1 / mean_eddy_turnover
    # np.set_printoptions(formatter={"all": np.format_float_scientific})
    # rounded_mean_eddy_turnover = np.round_(mean_eddy_turnover, decimals=2)

    print("mean_eddy_turnover", mean_eddy_turnover)

    fig, axes = plt.subplots(nrows=1, ncols=1)

    # axes[0].plot(
    #     np.log2(PAR.k_vec_temp), eddy_freq, "k.", label="Eddy freq. from $||u||$"
    # )
    # axes[0].plot(
    #     np.log2(PAR.k_vec_temp),
    #     (PAR.k_vec_temp / (2 * np.pi)) ** (2 / 3),
    #     "k--",
    #     label="$k^{2/3}$",
    # )

    # delta_time = (time[-1] - time[0]).real + PAR.stt

    # freq_title = g_plt_utils.generate_title(
    #     args,
    #     header_dict=header_dict,
    #     title_header="Eddy frequencies",
    #     title_suffix=f"$\\Delta t$={delta_time:.1f}s",
    # )

    # axes[0].set_yscale("log")
    # axes[0].grid()
    # axes[0].legend()
    # axes[0].set_xlabel("k")
    # axes[0].set_ylabel("Eddy frequency")
    # axes[0].set_title(
    #     freq_title,
    # )

    axes.plot(
        np.log2(PAR.k_vec_temp),
        mean_eddy_turnover,
        "k.",
        label="Eddy turnover time from $||u||$",
    )
    axes.plot(
        np.log2(PAR.k_vec_temp),
        (PAR.k_vec_temp / (2 * np.pi)) ** (-2 / 3),
        "k--",
        label="$k^{-2/3}$",
    )

    axes.set_xticks(SHELL_TICKS_COMPACT)
    axes.grid(False)

    # axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    time_title = g_plt_utils.generate_title(
        args,
        header_dict=header_dict,
        title_header="Eddy turnover time",
    )
    axes.set_yscale("log")
    axes.grid()
    axes.legend()
    axes.set_xlabel("$n$")
    axes.set_ylabel("$t_n$ [tu]")
    axes.set_title(time_title)

    if args["tolatex"]:
        plt_config.remove_legends(axes)
        plt_config.adjust_axes(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/appendices/timescale_analyses/",
            file_name="sh_eddy_turnover_times",
        )


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args
    # Initiate arrays
    # initiate_PAR.sdim_arrays(args["PAR.sdim"])
    # Initiate and update variables and arrays
    sh_utils.update_dependent_params(PAR)
    sh_utils.update_arrays(PAR)

    g_ui.confirm_run_setup(args)

    plt_config.adjust_default_fig_axes_settings(args)

    if "time_to_run" in args:
        args["Nt"] = int(args["time_to_run"] / PAR.dt * PAR.sample_rate)

    # Perform plotting
    if "energy" in args["plot_type"]:
        plot_energy(args=args)

    if "cum_energy" in args["plot_type"]:
        plot_energy_per_shell(args=args)

    if "energy_spectrum" in args["plot_type"]:
        plot_energy_spectrum(
            args=args,
        )
    if "vel_spectrum" in args["plot_type"]:
        # Import reference data
        _, _temp_u_data, _temp_header_dict = g_import.import_ref_data(args=args)

        plot_velocity_spectrum(
            _temp_u_data,
            _temp_header_dict,
            plot_arg_list=["kolmogorov", "calculate"],
            args=args,
        )

    if "pert_traj_energy_spectrum" in args["plot_type"]:
        plot_pert_traject_energy_spectrum(args)

    if "error_norm" in args["plot_type"]:
        if args["datapath"] is None:
            print("No path specified to analyse error norms.")
        else:
            g_plt_data.plot_error_norm_vs_time(
                args=args,
                legend_on=False,
                # cmap_list=["blue"],
                plot_args=[],
                normalize_start_time=False,
                raw_perturbations=True,
            )

    if "error_spectrum_vs_time" in args["plot_type"]:
        plot_error_energy_spectrum_vs_time_2D(args=args)

    if "shell_error" in args["plot_type"]:
        if args["datapath"] is None:
            print("No path specified to analyse shell error.")
        else:
            plot_shell_error_vs_time(args=args)

    if "eigen_mode_plot_3D" in args["plot_type"]:
        plot_3D_eigen_mode_analysis(args=args)

    if "eigen_mode_compare_3D" in args["plot_type"]:
        plot_3D_eigen_mode_anal_comparison(args=args)

    if "eigen_mode_plot_2D" in args["plot_type"]:
        plot_eigen_value_dist(args=args)

    if "eigen_vector_comp" in args["plot_type"]:
        plot_eigen_vector_comparison(args=args)

    if "error_vector_spectrogram" in args["plot_type"]:
        plot_error_vector_spectrogram(args=args)

    if "error_vector_spectrum" in args["plot_type"]:
        plot_error_vector_spectrum(args=args)

    if "energy_howmoller_rel_mean" in args["plot_type"]:
        plot_howmoller_diagram_u_energy(args=args, plt_args=["rel_mean"])

    if "hel_howmoller_rel_mean" in args["plot_type"]:
        plot_howmoller_diagram_helicity(args=args, plt_args=["rel_mean"])

    if "helicity_spectrum" in args["plot_type"]:
        _, u_data, header_dict = g_import.import_ref_data(args=args)
        plot_helicity_spectrum(u_data, header_dict, args)

    if "eddy_turnover" in args["plot_type"]:
        plot_eddie_turnover_times(args)

    g_plt_utils.save_or_show_plot(args)
