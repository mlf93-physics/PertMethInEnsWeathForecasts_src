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
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from lorentz63_experiments.params.params import *
import lorentz63_experiments.analyses.normal_mode_analysis as nm_analysis
import lorentz63_experiments.utils.util_funcs as l_utils
import general.utils.importing.import_data_funcs as g_import
import general.plotting.plot_data as g_plt_data
import general.utils.plot_utils as g_plt_utils
import general.utils.argument_parsers as a_parsers


def plot_attractor(args, ax=None):
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

    # Plot
    ax.plot3D(u_data[:, 0], u_data[:, 1], u_data[:, 2], "k-", alpha=0.6, linewidth=0.5)

    return time, u_data, header_dict


def plot_velocities(args):
    """Plot the velocities of the reference data

    Parameters
    ----------
    args : dict
        A dictionary containing run time arguments
    """

    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(time, u_data[:, i], "k-")
        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity")

    plt.suptitle("Velocities vs time")


def plot_energy(args, axes=None):
    """Plot the energy of the reference data

    Parameters
    ----------
    args : dict
        A dictionary containing run time arguments
    """

    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)
    energy = 1 / 2 * np.sum(u_data ** 2, axis=1)

    # Prepare axes
    if axes is None:
        axes = plt.gca()

    axes.plot(time, energy, "k-")
    axes.set_xlabel("Time")
    axes.set_ylabel("Energy")
    axes.set_title("Energy vs time")


def plot_error_norm_vs_time(args):

    g_plt_data.plot_error_norm_vs_time(args)


def plot_normal_mode_dist(args):

    (
        u_profiles,
        e_values,
        e_vector_matrix,
        ref_header_dict,
    ) = nm_analysis.analyse_normal_mode_dist(args)

    # Setup axes
    fig1 = plt.figure()
    ax1 = plt.axes(projection="3d")

    max_e_value = np.max(e_values.real)
    min_e_value = np.min(e_values.real)

    # Prepare cmap and norm
    cmap, norm = g_plt_utils.get_cmap_distributed_around_zero(
        vmin=min_e_value, vmax=max_e_value
    )
    # Plot attractor
    plot_attractor(args, ax=ax1)

    # Plot
    scatter_plot = ax1.scatter(
        u_profiles[0, :],
        u_profiles[1, :],
        u_profiles[2, :],
        c=e_values.real,
        norm=norm,
        cmap=cmap,
    )

    e_value_dist_title = g_plt_utils.generate_title(
        ref_header_dict,
        args,
        title_header="Eigen value dist | Lorentz63 model \n",
        title_suffix=f"$N_{{points}}$={args['n_profiles']}",
    )

    ax1.set_title(e_value_dist_title)
    fig1.colorbar(scatter_plot)

    fig2 = plt.figure()
    ax2 = plt.axes(projection="3d")
    plot_attractor(args, ax=ax2)
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
        length=2,
    )

    e_vector_dist_title = g_plt_utils.generate_title(
        ref_header_dict,
        args,
        title_header="Eigen vector dist colored by eigen values | Lorentz63 model \n",
        title_suffix=f"$N_{{points}}$={args['n_profiles']}",
    )

    ax2.set_title(e_vector_dist_title)

    # Set quiver colors
    qplot.set_array(np.concatenate((e_values.real, np.repeat(e_values.real, 2))))
    # Set colorbar
    fig2.colorbar(qplot)

    if args["save_fig"]:
        out_path = pl.Path(
            "../../thesis/figures/lorentz63_experiments/normal_mode_perturbations/"
        )
        name1 = f"nm_perturbations_e_value_dist_n{args['n_profiles']}"
        name2 = f"nm_perturbations_e_vector_dist_n{args['n_profiles']}"
        g_plt_utils.save_interactive_fig(fig1, out_path, name1)
        g_plt_utils.save_interactive_fig(fig2, out_path, name2)


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
        ref_header_dict, args, title_header="E dist | Lorentz63 model \n"
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

    # Calculate deriv_matrix
    deriv_matrix = l_utils.setup_deriv_matrix(args)
    deriv_matrix_array = np.repeat(
        np.reshape(deriv_matrix, (1, sdim, sdim)), u_data.shape[0], axis=0
    )
    deriv_matrix_array[:, 1, 2] = u_data[:, 0]
    deriv_matrix_array[:, 2, 0] = u_data[:, 1]

    # Calculate du_array
    du_data = deriv_matrix_array @ np.reshape(u_data, (*u_data.shape, 1))
    du_data = np.squeeze(du_data, axis=2)

    # Calculate change in energy
    dE_array = np.sum(du_data * u_data, axis=1)

    # Prepare cmap and norm
    cmap, norm = g_plt_utils.get_cmap_distributed_around_zero(
        vmin=np.min(dE_array), vmax=np.max(dE_array)
    )

    coll2.set_array((dE_array[1:] + dE_array[:-1]) / 2)

    de_dist_title = g_plt_utils.generate_title(
        ref_header_dict, args, title_header="dE dist | Lorentz63 model \n"
    )

    ax2.add_collection(coll2)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax2.set_zlim(0, 40)
    ax2.grid(False)
    # ax2.set_facecolor((1, 1, 1))
    ax2.set_title(de_dist_title)
    fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2)


if __name__ == "__main__":
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args
    print("args", args)

    if "time_to_run" in args:
        args["Nt"] = int(args["time_to_run"] / dt * sample_rate)

    if "attractor_plot" in args["plot_type"]:
        plot_attractor(args)
    elif "velocity_plot" in args["plot_type"]:
        plot_velocities(args)
    elif "energy_plot" in args["plot_type"]:
        plot_energy(args)
    elif "error_norm" in args["plot_type"]:
        plot_error_norm_vs_time(args)
    elif "nm_dist" in args["plot_type"]:
        plot_normal_mode_dist(args)
    elif "energy_dist" in args["plot_type"]:
        plot_energy_dist(args)

    if not args["noplot"]:
        plt.tight_layout()
        plt.show()
