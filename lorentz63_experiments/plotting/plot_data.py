import sys

sys.path.append("..")
import argparse
from mpl_toolkits import mplot3d
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from lorentz63_experiments.params.params import *
import lorentz63_experiments.analyses.normal_mode_analysis as nm_analysis
import general.utils.import_data_funcs as g_import
import general.plotting.plot_data as g_plt_data


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

    u_profiles, e_values = nm_analysis.analyse_normal_mode_dist(args)

    # Setup axes
    ax = plt.axes(projection="3d")

    max_e_value = np.max(e_values.real)

    # Plot
    # for i in range(u_profiles.shape[1]):
    scatter_plot = ax.scatter(
        u_profiles[0, :],
        u_profiles[1, :],
        u_profiles[2, :],
        c=e_values.real,
        norm=mpl_colors.Normalize(-max_e_value, max_e_value),
        cmap="coolwarm",
    )
    plt.colorbar(scatter_plot)


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", nargs="?", default=None, type=str)
    arg_parser.add_argument("--plot_type", nargs="+", default=None, type=str)
    """
    plot_mode :
        standard : plot everything with the standard plot setup
        detailed : plot extra details in plots
    """
    arg_parser.add_argument("--plot_mode", nargs="?", default="standard", type=str)
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    arg_parser.add_argument("--start_time", nargs="+", type=float)
    arg_parser.add_argument("--specific_ref_records", nargs="+", default=[0], type=int)
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--n_profiles", default=1, type=int)
    arg_parser.add_argument("--n_runs_per_profile", default=1, type=int)
    arg_parser.add_argument("--time_to_run", default=0.1, type=float)
    arg_parser.add_argument("--ref_start_time", default=0, type=float)
    arg_parser.add_argument("--ref_end_time", default=-1, type=float)
    arg_parser.add_argument("--xlim", nargs=2, default=None, type=float)
    arg_parser.add_argument("--ylim", nargs=2, default=None, type=float)
    arg_parser.add_argument("--sigma", default=10, type=float)
    arg_parser.add_argument("--r_const", default=28, type=float)
    arg_parser.add_argument("--b_const", default=8 / 3, type=float)

    subparsers = arg_parser.add_subparsers()
    perturb_parser = subparsers.add_parser(
        "perturb_plot",
        help="Arguments needed for plotting the perturbation vs time plot.",
    )
    perturb_parser.add_argument("--perturb_folder", nargs="?", default=None, type=str)
    perturb_parser.add_argument("--n_files", default=np.inf, type=int)
    perturb_parser.add_argument("--file_offset", default=0, type=int)
    perturb_parser.add_argument("--specific_files", nargs="+", default=None, type=int)
    perturb_parser.add_argument("--endpoint", action="store_true")
    perturb_parser.add_argument("--combinations", action="store_true")

    args = vars(arg_parser.parse_args())
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

    plt.show()
