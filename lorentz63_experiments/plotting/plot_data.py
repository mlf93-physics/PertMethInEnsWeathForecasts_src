import sys

sys.path.append("..")
import argparse
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import general.utils.import_data_funcs as g_import
from lorentz63_experiments.params.params import *


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


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", nargs="?", default=None, type=str)
    arg_parser.add_argument("--plot_type", nargs="+", default=None, type=str)
    """
    plot_mode :
        standard : plot everything with the standard plot setup
        detailted : plot extra details in plots
    """
    arg_parser.add_argument("--plot_mode", nargs="?", default="standard", type=str)
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    arg_parser.add_argument("--start_time", nargs="+", type=float)
    arg_parser.add_argument("--specific_ref_records", nargs="+", default=[0], type=int)

    subparsers = arg_parser.add_subparsers()
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--n_profiles", default=1, type=int)
    arg_parser.add_argument("--n_runs_per_profile", default=1, type=int)
    arg_parser.add_argument("--time_to_run", default=0.1, type=float)
    arg_parser.add_argument("--ref_start_time", default=0, type=float)
    arg_parser.add_argument("--ref_end_time", default=-1, type=float)

    perturb_parser = subparsers.add_parser(
        "perturb_plot",
        help="Arguments needed for plotting the perturbation vs time plot.",
    )
    perturb_parser.add_argument("--perturb_folder", nargs="?", default=None, type=str)
    perturb_parser.add_argument("--n_files", default=np.inf, type=int)
    perturb_parser.add_argument("--file_offset", default=0, type=int)
    perturb_parser.add_argument("--specific_files", nargs="+", default=None, type=int)
    perturb_parser.add_argument("--endpoint", action="store_true")

    args = vars(arg_parser.parse_args())
    print("args", args)

    # Set seed if wished
    if args["seed_mode"]:
        np.random.seed(seed=1)

    if "burn_in_time" in args:
        args["burn_in_lines"] = int(args["burn_in_time"] / dt * sample_rate)
    if "time_to_run" in args:
        args["Nt"] = int(args["time_to_run"] / dt * sample_rate)

    if "attractor" in args["plot_type"]:
        plot_attractor(args)

    plt.show()
