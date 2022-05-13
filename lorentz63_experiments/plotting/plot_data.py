"""Plotting functions relevant for the Lorentz63 model experiments

Example
-------
python plotting/plot_data.py --plot_type=error_norm --exp_folder=test_breed_vector_eof

"""

import sys

sys.path.append("..")
import pathlib as pl
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.signal as sp_signal
from lorentz63_experiments.params.params import *
import lorentz63_experiments.analyses.normal_mode_analysis as nm_analysis
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
import lorentz63_experiments.utils.util_funcs as l_utils
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
from general.plotting.plot_params import *
import general.plotting.plot_data as g_plt_data
import general.analyses.plot_analyses as g_plt_anal
import general.utils.plot_utils as g_plt_utils
import general.utils.util_funcs as g_utils
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
import general.plotting.plot_config as plt_config
import sklearn.cluster as skl_cluster
import scipy.optimize as sp_optim
import config as cfg


def plot_splitted_wings(args):
    fig1, axes = plt.subplots(nrows=1, ncols=2, facecolor=(0, 0, 0, 0), sharey=True)

    # Import reference data
    _, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    wing_indices1 = u_data[:, 0] > 0
    wing_indices2 = np.logical_not(wing_indices1)

    segments1 = zip(u_data[:-1, :2], u_data[1:, :2])
    segments2 = zip(u_data[:-1, 2:0:-1], u_data[1:, 2:0:-1])

    coll1 = LineCollection(segments1, cmap="coolwarm")
    coll1.set_array(wing_indices1)
    coll2 = LineCollection(segments2, cmap="coolwarm")
    coll2.set_array(wing_indices1)

    e_dist_title = g_plt_utils.generate_title(
        args, header_dict=ref_header_dict, title_header="E dist | Lorentz63 model \n"
    )

    line_plot1 = axes[0].add_collection(coll1)
    axes[0].set_xlim(-20, 20)
    axes[0].set_ylim(-25, 25)
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].grid(False)

    line_plot2 = axes[1].add_collection(coll2)
    axes[1].set_xlim(0, 50)
    # axes[1].set_ylim(-25, 25)
    axes[1].set_xlabel("$z$")
    axes[1].grid(False)

    if args["tolatex"]:
        plt_config.adjust_axes(axes)
        g_plt_utils.add_subfig_labels(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath=pl.Path("thesis_figures/pt_methods/"),
            file_name="l63_attractor_split",
        )


def plot_attractor_standalone(args, ax=None, alpha=1):
    """Plot the 3D attractor of the reference data

    Parameters
    ----------
    args : dict
        A dictionary containing run time arguments
    """
    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    # Setup axis if necessary
    if ax is None:
        ax = plt.axes(projection="3d")

    plot_style = "k-"
    linewidth = 0.5

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    # Plot
    ax.plot3D(
        u_data[:, 0],
        u_data[:, 1],
        u_data[:, 2],
        plot_style,
        alpha=alpha,
        linewidth=linewidth,
        zorder=10,
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.view_init(elev=13, azim=-45)

    ax.plot(
        np.sqrt(args["b_const"] * (args["r_const"] - 1)),
        np.sqrt(args["b_const"] * (args["r_const"] - 1)),
        args["r_const"] - 1,
        "k.",
    )

    ds = 1
    ax.plot(
        -np.sqrt(args["b_const"] * (args["r_const"] - 1)),
        -np.sqrt(args["b_const"] * (args["r_const"] - 1)),
        args["r_const"] - 1,
        "k.",
    )

    # Add text in +/- fixpoints
    ax.text(
        np.sqrt(args["b_const"] * (args["r_const"] - 1)) + ds,
        np.sqrt(args["b_const"] * (args["r_const"] - 1)) + ds,
        args["r_const"] - 1,
        "R",
    )
    ax.text(
        -np.sqrt(args["b_const"] * (args["r_const"] - 1)) + ds,
        -np.sqrt(args["b_const"] * (args["r_const"] - 1)) + ds,
        args["r_const"] - 1,
        "L",
    )
    ax.text(
        np.sqrt(args["b_const"] * (args["r_const"] - 1)) - ds - 3,
        np.sqrt(args["b_const"] * (args["r_const"] - 1)) - ds,
        args["r_const"] - 1 - 3,
        "$\\mathbf{x}_+$",
    )
    ax.text(
        -np.sqrt(args["b_const"] * (args["r_const"] - 1)) - ds - 3,
        -np.sqrt(args["b_const"] * (args["r_const"] - 1)) - ds,
        args["r_const"] - 1 - 3,
        "$\\mathbf{x}_-$",
    )

    if args["tolatex"]:
        plt_config.adjust_axes(ax)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath=pl.Path("thesis_figures/models"),
            file_name="lorentz_attractor",
        )

    return time, u_data, header_dict


def plot_attractor(args, ax=None, alpha=1):
    """Plot the 3D attractor of the reference data

    Parameters
    ----------
    args : dict
        A dictionary containing run time arguments
    """
    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    # Setup axis if necessary
    if ax is None:
        ax = plt.axes(projection="3d")

    wing_indices1 = u_data[:, 0] > 0
    wing_indices2 = np.logical_not(wing_indices1)

    # split_attractor_in_wings(u_data, axes=ax)

    if "ref_highlight" in args:
        if args["ref_highlight"]:
            plot_style = "r-"
            linewidth = 2

            # Add start_point
            ax.plot3D(
                u_data[0, 0],
                u_data[0, 1],
                u_data[0, 2],
                "ro",
                zorder=10,
            )
    else:
        plot_style = "k-"
        linewidth = 0.5

    # Plot
    ax.plot3D(
        u_data[:, 0],
        u_data[:, 1],
        u_data[:, 2],
        plot_style,
        alpha=alpha,
        linewidth=linewidth,
        zorder=10,
    )

    if "ref_highlight" in args:
        if args["ref_highlight"]:
            # Plot some of the attractor in the background
            args["ref_start_time"] = args["ref_start_time"] - 50
            args["ref_end_time"] = args["ref_end_time"] + 50
            time, u_data, header_dict = g_import.import_ref_data(args=args)

            plot_style = "k-"

            ax.plot3D(
                u_data[:, 0],
                u_data[:, 1],
                u_data[:, 2],
                plot_style,
                alpha=1,
                linewidth=0.5,
                zorder=0,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return time, u_data, header_dict


def split_attractor_in_wings(u_data, axes=None):

    kmeans_clf = skl_cluster.KMeans(
        n_clusters=2, n_init=10, max_iter=3000, tol=1e-04, random_state=0
    )
    km_prediction = kmeans_clf.fit_predict(u_data[:, [0, 1]])

    axes.plot3D(
        u_data[km_prediction == 0, 0],
        u_data[km_prediction == 0, 1],
        u_data[km_prediction == 0, 2],
        "b.",
        alpha=0.6,
    )

    axes.plot3D(
        u_data[km_prediction == 1, 0],
        u_data[km_prediction == 1, 1],
        u_data[km_prediction == 1, 2],
        "r.",
        alpha=0.6,
    )


def plot_velocities(args, axes=None):
    """Plot the velocities of the reference data

    Parameters
    ----------
    args : dict
        A dictionary containing run time arguments
    """

    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1)

    # for i, ax in enumerate(axes):
    axes.plot(time - time[0], u_data[:, 0], "k-")
    axes.set_xlabel("Time [tu]")
    axes.set_ylabel("x")
    axes.set_ylim(-20, 20)

    if args["tolatex"]:
        plt_config.adjust_axes([axes])

    if args["save_fig"]:
        out_path = pl.Path("thesis_figures/models/")
        g_plt_utils.save_figure(
            args,
            fig=fig,
            subpath=out_path,
            file_name="l63_xvelocity_vs_time",
        )


def plot_energy(args, axes=None, zorder=0, alpha=1):
    """Plot the energy of the reference data

    Parameters
    ----------
    args : dict
        A dictionary containing run time arguments
    """

    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)
    energy = u_data[:, 0]  # 1 / 2 * np.sum(u_data ** 2, axis=1)

    # Prepare axes
    if axes is None:
        axes = plt.gca()

    axes.plot(
        time, energy, "k-", zorder=zorder, alpha=alpha, linewidth=LINEWIDTHS["thin"]
    )
    axes.set_xlabel("Time")
    axes.set_ylabel("Energy, $\\frac{1}{2}\\sum_i {x}_{i, ref}$")
    axes.set_title("Energy vs time")


def plot_error_norm_vs_time(args):

    g_plt_data.plot_error_norm_vs_time(
        args,
        normalize_start_time=False,
        legend_on=False,
        cmap_list=["blue"],
        plot_args=["unique_linestyle"],
        linear_fit=False,
    )


def plot_normal_mode_dist(args):

    (
        u_profiles,
        e_values,
        e_vector_matrix,
        e_vector_collection,
        ref_header_dict,
    ) = nm_analysis.analyse_normal_mode_dist(args)

    part = "real"
    if part == "real":
        chosen_e_values = e_values.real
    else:
        chosen_e_values = e_values.imag

    orthonormality_matrix = 0
    for item in e_vector_collection:
        temp = g_plt_anal.orthogonality_of_vectors(item.T)
        # print("temp", temp)
        orthonormality_matrix += temp  # g_plt_anal.orthogonality_of_vectors(item.T)
    orthonormality_matrix /= len(e_vector_collection)

    print("Orthogonality of eigen vectors: ", orthonormality_matrix)

    # Setup axes
    fig1 = plt.figure()
    ax1 = plt.axes(projection="3d")
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    max_e_value = np.max(chosen_e_values)
    min_e_value = np.min(chosen_e_values)

    # Prepare cmap and norm
    cmap, norm = g_plt_utils.get_custom_cmap(
        vmin=min_e_value,
        vmax=max_e_value,
        vcenter=(max_e_value + min_e_value) / 2,
        cmap_handle=plt.cm.jet if part == "real" else plt.cm.jet,
    )
    # Plot attractor
    plot_attractor(args, ax=ax1, alpha=0.5)

    # Plot
    scatter_plot = ax1.scatter(
        u_profiles[0, :],
        u_profiles[1, :],
        u_profiles[2, :],
        c=chosen_e_values,
        norm=norm,
        cmap=cmap,
    )

    e_value_dist_title = g_plt_utils.generate_title(
        args,
        header_dict=ref_header_dict,
        title_header="Eigen value dist | Lorentz63 model \n",
        title_suffix=f"$N_{{points}}$={args['n_profiles']}",
    )

    ax1.set_title(e_value_dist_title)
    ax1.grid(False)
    fig1.colorbar(scatter_plot)

    fig2 = plt.figure(figsize=(5.39749 / 2 + 5.39749 / 5, 4.1))
    ax2 = plt.axes(projection="3d")
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plot_attractor(args, ax=ax2, alpha=0.2)
    qplot = ax2.quiver(
        u_profiles[0, :],
        u_profiles[1, :],
        u_profiles[2, :],
        e_vector_matrix[0, :].real,
        e_vector_matrix[1, :].real,
        e_vector_matrix[2, :].real,
        norm=norm,
        cmap=cmap,
        normalize=True,
        alpha=1,
        length=2,
        linewidths=0.3,
        zorder=10,
    )

    ax2.set_xlabel("$x$", labelpad=-8)
    ax2.set_ylabel("$y$", labelpad=-8)
    ax2.set_zlabel("$z$", labelpad=-12)
    ax2.set_xticks([-20, 0, 20])
    ax2.set_yticks([-20, 0, 20])
    ax2.tick_params(axis="both", which="major", pad=-3)

    e_vector_dist_title = g_plt_utils.generate_title(
        args,
        header_dict=ref_header_dict,
        title_header="Eigen vector dist colored by eigen values | Lorentz63 model \n",
        title_suffix=f"$N_{{points}}$={args['n_profiles']}",
    )

    # ax2.set_title(e_vector_dist_title)
    ax2.grid(False)

    # Set quiver colors
    qplot.set_array(np.concatenate((chosen_e_values, np.repeat(chosen_e_values, 2))))
    # Set colorbar
    if part == "real":
        label = "$\\Re(\\mu_1)$"
    elif part == "imag":
        label = "$\\Im(\\mu_1)$"

    fig2.colorbar(qplot, shrink=0.6, pad=-0.05, location="bottom", label=label)
    ax2.view_init(elev=13, azim=-45)
    fig2.subplots_adjust(
        top=1.0, bottom=0.105, left=0.015, right=0.924, hspace=0.175, wspace=0.18
    )

    if args["tolatex"]:
        plt_config.adjust_axes([ax2])

    if args["save_fig"]:
        out_path = pl.Path("thesis_figures/models/")
        g_plt_utils.save_figure(
            args,
            fig=fig2,
            subpath=out_path,
            file_name=args["save_fig_name"],
        )
        # g_plt_utils.save_interactive_fig(fig1, out_path, name1)
        # g_plt_utils.save_interactive_fig(fig2, out_path, name2)


def plot_energy_dist(args):

    # Plot energy distribution
    fig1 = plt.figure()
    ax1 = plt.axes(projection="3d", facecolor=(0, 0, 0, 0))

    # Import reference data
    _, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    segments = zip(u_data[:-1, :], u_data[1:, :])

    coll1 = Line3DCollection(segments, cmap="coolwarm")
    coll1.set_array(
        (np.sum(u_data[1:, :] ** 2, axis=1) + np.sum(u_data[:-1, :] ** 2, axis=1)) / 2
    )

    e_dist_title = g_plt_utils.generate_title(
        args, header_dict=ref_header_dict, title_header="E dist | Lorentz63 model \n"
    )

    line_plot1 = ax1.add_collection(coll1)
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.set_zlim(0, 40)
    ax1.grid(False)
    # ax1.set_facecolor((1, 1, 1))
    ax1.set_title(e_dist_title)
    fig1.colorbar(line_plot1)

    # Plot "change in energy" distribution

    fig2 = plt.figure()
    ax2 = plt.axes(projection="3d", facecolor=(0, 0, 0, 0))

    segments = zip(u_data[:-1, :], u_data[1:, :])

    coll2 = Line3DCollection(segments, cmap="coolwarm")

    # Calculate lorentz_matrix
    lorentz_matrix = l_utils.setup_lorentz_matrix(args)
    lorentz_matrix_array = np.repeat(
        np.reshape(lorentz_matrix, (1, sdim, sdim)), u_data.shape[0], axis=0
    )
    lorentz_matrix_array[:, 1, 2] = u_data[:, 0]
    lorentz_matrix_array[:, 2, 0] = u_data[:, 1]

    # Calculate du_array
    du_data = lorentz_matrix_array @ np.reshape(u_data, (*u_data.shape, 1))
    du_data = np.squeeze(du_data, axis=2)

    # Calculate change in energy
    dE_array = np.sum(du_data * u_data, axis=1)

    # Prepare cmap and norm
    cmap, norm = g_plt_utils.get_custom_cmap(
        vmin=np.min(dE_array), vmax=np.max(dE_array)
    )

    coll2.set_array((dE_array[1:] + dE_array[:-1]) / 2)

    de_dist_title = g_plt_utils.generate_title(
        args, header_dict=ref_header_dict, title_header="dE dist | Lorentz63 model \n"
    )

    ax2.add_collection(coll2)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax2.set_zlim(0, 40)
    ax2.grid(False)
    # ax2.set_facecolor((1, 1, 1))
    ax2.set_title(de_dist_title)
    fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2)


def plot_characteristic_periods(args: dict, axes: plt.Axes = None):
    # Import reference data
    time, u_data, ref_header_dict = g_import.import_ref_data(args=args)
    # variable = 1 / 2 * np.sum(u_data ** 2, axis=1)
    variable = u_data[:, 2]

    spectrum = np.fft.fft(variable)
    power_spec = np.abs(spectrum) ** 2
    freqs = np.fft.fftfreq(variable.shape[0], d=stt)
    idx = np.argsort(freqs)[(freqs.size // 2 + 1) :]

    # Prepare axes
    if axes is None:
        fig, axes = plt.subplots()

    power_spec /= np.max(power_spec[idx])

    peaks = sp_signal.find_peaks(power_spec[idx], height=1e-1)
    max_height_index = np.argmax(peaks[1]["peak_heights"])
    print("Rotation time: ", 1 / freqs[peaks[0][max_height_index]])

    axes.plot(1 / freqs[idx], power_spec[idx], "k", linewidth=0.2)
    axes.set_xscale("log")
    axes.set_yscale("log")
    # axes.set_yticks([1e-6, 1e-3, 1])
    # axes.set_yticklabels([1e-9, 1e-6, 1e-3, 1])
    axes.set_ylabel("Power")
    axes.set_xlabel("Period [tu]")
    axes.set_title("Power spectrum | y")
    axes.grid(False)

    if args["tolatex"]:
        plt_config.adjust_axes([axes])

    if args["save_fig"]:
        out_path = pl.Path("thesis_figures/appendices/timescale_analyses/")
        g_plt_utils.save_figure(
            args,
            fig=fig,
            subpath=out_path,
            file_name="power_spectrum_vs_period_var_z",
        )


def plot_residence_time_in_wing(args):
    # Import reference data
    time, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    wing_indices = u_data[:, 0] > 0

    roll_array = np.roll(wing_indices, 1)
    neg_to_pos_wing_boolean_array = np.logical_and(
        np.logical_not(roll_array), wing_indices
    )
    pos_to_neg_wing_boolean_array = np.logical_and(
        roll_array, np.logical_not(wing_indices)
    )

    # Convert indices to start times
    wing_times = []
    wing_times.append(time[np.where(neg_to_pos_wing_boolean_array)])
    wing_times.append(time[np.where(pos_to_neg_wing_boolean_array)])

    # Find largest first time
    largest_first_time_index = np.argmin([wing_times[0][0], wing_times[1][0]])

    # Calculate residence times
    residence_times1 = (
        wing_times[largest_first_time_index][1:]
        - wing_times[abs(largest_first_time_index - 1)][:-1]
    )
    # residence_times2 = (
    #     wing_times[abs(largest_first_time_index - 1)][:-1]
    #     - wing_times[largest_first_time_index][:-1]
    # )
    # Concatenate residence times
    residence_times = np.abs(
        residence_times1
    )  # np.abs(np.concatenate([residence_times1, residence_times2]))
    fig, axes = plt.subplots(nrows=1, ncols=1)
    values, bins, bars = axes.hist(
        residence_times, bins=20, density=True, edgecolor="k", fc=(0, 0, 0, 0.5)
    )

    def linearfunction(x, a):
        return a * x

    # Sort out zeros
    non_zero_values = values != 0
    bin_centers = 1 / 2 * (bins[:-1] + bins[1:])
    log_norm_data = np.log(values, where=non_zero_values)
    lin_popt, lin_pcov = sp_optim.curve_fit(
        linearfunction,
        bin_centers[non_zero_values],
        log_norm_data[non_zero_values],
    )

    residence_time = -1 / lin_popt[0]

    print("residence_time", residence_time)
    print("lin_popt", lin_popt)

    axes.set_yscale("log")

    # Plot fit
    axes.plot(
        bin_centers,
        np.exp(linearfunction(bin_centers, *lin_popt)),
        color="k",
        linestyle="dashed",
    )

    title = g_plt_utils.generate_title(
        args,
        header_dict=ref_header_dict,
        title_header="Wing residence time histogram",
    )

    axes.set_title(title)
    axes.set_xlabel("Residence time [tu]")
    axes.set_ylabel("Frequency")

    if args["tolatex"]:
        plt_config.adjust_axes([axes])

    if args["save_fig"]:
        out_path = pl.Path("thesis_figures/appendices/timescale_analyses/")
        g_plt_utils.save_figure(
            args,
            fig=fig,
            subpath=out_path,
            file_name="residence_time_in_wing_histogram",
        )

    # plt.figure()
    # plt.plot(time, u_data[:, 0])
    # plt.plot(
    #     wing_times[largest_first_time_index],
    #     np.zeros(len(wing_times[largest_first_time_index])),
    #     "x",
    #     color="b",
    #     label="index1",
    # )
    # plt.plot(
    #     wing_times[abs(largest_first_time_index - 1)],
    #     np.zeros(len(wing_times[abs(largest_first_time_index - 1)])),
    #     "x",
    #     color="r",
    #     label="index2",
    # )
    # plt.legend()

    if args["tolatex"]:
        plt_config.adjust_axes([axes])

    if args["save_fig"]:
        out_path = pl.Path("thesis_figures/appendices/timescale_analyses/")
        g_plt_utils.save_figure(
            args,
            fig=fig,
            subpath=out_path,
            file_name="power_spectrum_vs_period_var_z",
        )


def projectibility_bv_eof_vs_nm(args):

    (
        vector_units,
        _,
        u_init_profiles,
        eval_pos,
        perturb_header_dicts,
    ) = pt_import.import_perturb_vectors(
        args,
        raw_perturbations=True,
    )

    (
        e_vector_matrix,
        e_values_max,
        e_vector_collection,
        e_value_collection,
    ) = l63_nm_estimator.find_normal_modes(
        u_init_profiles, args, n_profiles=u_init_profiles.shape[1]
    )

    # Normalize
    vector_units = g_utils.normalize_array(vector_units, norm_value=1, axis=2)
    e_vector_collection = g_utils.normalize_array(
        np.array(e_vector_collection), norm_value=1, axis=2
    )

    projectibility = np.zeros((args["n_runs_per_profile"], args["n_runs_per_profile"]))
    for i in range(args["n_profiles"]):
        for j in range(args["n_runs_per_profile"]):
            # projectibility[j, :] += g_plt_anal.orthogonality_to_vector(
            #     vector_units[i, j, :], e_vector_matrix.T
            # )

            projectibility[j, :] += np.abs(
                g_plt_anal.orthogonality_to_vector(
                    vector_units[i, j, :], e_vector_collection[i, :, :].T
                )
            )

    sb.heatmap(projectibility / args["n_profiles"], cmap="Reds", annot=True)
    plt.xlabel("NM index")
    plt.ylabel("BV-EOF index")


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args

    # Initiate arrays
    # initiate_sdim_arrays(args["sdim"])
    g_ui.confirm_run_setup(args)
    plt_config.adjust_default_fig_axes_settings(args)

    if "time_to_run" in args:
        args["Nt"] = int(args["time_to_run"] / dt * sample_rate)

    if "attractor" in args["plot_type"]:
        plot_attractor_standalone(args)
    elif "velocity" in args["plot_type"]:
        plot_velocities(args)
    elif "energy" in args["plot_type"]:
        plot_energy(args)
    elif "error_norm" in args["plot_type"]:
        plot_error_norm_vs_time(args)
    elif "nm_dist" in args["plot_type"]:
        plot_normal_mode_dist(args)
    elif "energy_dist" in args["plot_type"]:
        plot_energy_dist(args)
    elif "periods" in args["plot_type"]:
        plot_characteristic_periods(args)
    elif "residence_time" in args["plot_type"]:
        plot_residence_time_in_wing(args)
    elif "splitted_wings" in args["plot_type"]:
        plot_splitted_wings(args)
    elif "project_bv_eof_vs_nm" in args["plot_type"]:
        projectibility_bv_eof_vs_nm(args)

    g_plt_utils.save_or_show_plot(args)
