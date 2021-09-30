import sys

sys.path.append("..")
from mpl_toolkits import mplot3d
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from lorentz63_experiments.params.params import *
import lorentz63_experiments.analyses.normal_mode_analysis as nm_analysis
import general.utils.importing.import_data_funcs as g_import
import general.plotting.plot_data as g_plt_data
import general.utils.plot_utils as g_plt_utils
import general.utils.argument_parsers as a_parsers


def plot_attractor(args):
    """Plot the 3D attractor of the reference data

    Parameters
    ----------
    args : dict
        A dictionary containing run time arguments
    """
    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    # Setup axes
    ax = plt.axes(projection="3d")

    # Plot
    ax.plot3D(u_data[:, 0], u_data[:, 1], u_data[:, 2], "k-", linewidth=0.5)


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


def plot_energy(args):
    """Plot the energy of the reference data

    Parameters
    ----------
    args : dict
        A dictionary containing run time arguments
    """

    # Import reference data
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    energy = 1 / 2 * np.sum(u_data ** 2, axis=1)
    plt.plot(time, energy, "k-")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy vs time")


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

    cmap, norm = g_plt_utils.get_cmap_distributed_around_zero(
        vmin=-max_e_value, vmax=max_e_value
    )

    # print("num pos e values: ", np.sum(e_values.real > 0))
    # print("num neg e values: ", np.sum(e_values.real < 0))

    # Plot
    scatter_plot = ax1.scatter(
        u_profiles[0, :],
        u_profiles[1, :],
        u_profiles[2, :],
        c=e_values.real,
        norm=norm,  # mpl_colors.Normalize(-max_e_value, max_e_value),
        cmap=cmap,
    )
    ax1.set_title(
        "Eigen value dist | Lorentz63 model \n"
        + f"$N_{{points}}$={args['n_profiles']}, $\\sigma={ref_header_dict['sigma']}$"
        + f", r={ref_header_dict['r_const']}, b={ref_header_dict['b_const']:.2f}"
    )
    fig1.colorbar(scatter_plot)

    fig2 = plt.figure()
    cmap = "coolwarm"
    ax2 = plt.axes(projection="3d")
    qplot = ax2.quiver(
        u_profiles[0, :],
        u_profiles[1, :],
        u_profiles[2, :],
        e_vector_matrix[0, :].real,
        e_vector_matrix[1, :].real,
        e_vector_matrix[2, :].real,
        norm=mpl_colors.Normalize(-max_e_value, max_e_value),
        cmap=cmap,
        normalize=True,
        length=2,
    )
    ax2.set_title(
        "Eigen vector dist colored by eigen values | Lorentz63 model \n"
        + f"$N_{{points}}$={args['n_profiles']}, $\\sigma={ref_header_dict['sigma']}$"
        + f", r={ref_header_dict['r_const']}, b={ref_header_dict['b_const']:.2f}"
    )

    qplot.set_array(np.linspace(-max_e_value, max_e_value, 100))
    # Prepare quiver colors
    colors = e_values.real
    # Flatten and normalize
    colors = (colors.ravel()) / (0.5 * colors.ptp())

    # Repeat for each body line and two head lines
    colors = np.concatenate((colors, np.repeat(colors, 2)))
    # repeated_mask = np.concatenate((mask.ravel(), np.repeat(mask.ravel(), 2)))
    # Colormap
    colors = getattr(plt.cm, cmap)(colors)
    qplot.set_edgecolor(colors)
    qplot.set_facecolor(colors)
    fig2.colorbar(qplot)


if __name__ == "__main__":
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = vars(stand_plot_arg_parser.args)
    print("args", args)

    # Set seed if wished
    if args["seed_mode"]:
        np.random.seed(seed=1)

    if "burn_in_time" in args:
        args["burn_in_lines"] = int(args["burn_in_time"] / dt * sample_rate)
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

    if not args["noplot"]:
        plt.tight_layout()
        plt.show()
