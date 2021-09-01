import os
from math import log10
import argparse
from pathlib import Path
import numpy as np
import multiprocessing
from pyinstrument import Profiler
from sabra_model.sabra_model import run_model
from params.params import *
from utils.save_data_funcs import save_data, save_perturb_info
from utils.import_data_funcs import import_start_u_profiles
from lyaponov.lyaponov_exp_estimator import (
    find_eigenvector_for_perturbation,
    calculate_perturbations,
)


profiler = Profiler()


def perturbation_runner(
    u_old, perturb_positions, du_array, data_out, args, run_count, perturb_count
):
    """Execute the sabra model on one given perturbed u_old profile"""

    print(
        f"Running perturbation {run_count + 1}/"
        + f"{args['n_profiles']*args['n_runs_per_profile']} | profile"
        + f" {run_count // args['n_runs_per_profile']}, profile run"
        + f" {run_count % args['n_runs_per_profile']}"
    )

    run_model(
        u_old,
        du_array,
        data_out,
        args["Nt"] + args["endpoint"] * 1,
        args["ny"],
        args["forcing"],
    )
    save_data(
        data_out,
        subfolder=Path(args["path"]).name,
        prefix=f"perturb{perturb_count}_",
        perturb_position=perturb_positions[run_count // args["n_runs_per_profile"]],
        args=args,
    )


def main(args=None):
    args["Nt"] = int(args["time_to_run"] / dt)
    args["burn_in_lines"] = int(args["burn_in_time"] / dt * sample_rate)

    # Import start profiles
    u_init_profiles, perturb_positions, header_dict = import_start_u_profiles(args=args)

    # Save parameters to args dict:
    args["forcing"] = header_dict["f"].real

    if args["ny_n"] is None:
        args["ny"] = header_dict["ny"]

        if args["forcing"] == 0:
            args["ny_n"] = 0
        else:
            args["ny_n"] = int(
                3
                / 8
                * log10(args["forcing"] / (header_dict["ny"] ** 2))
                / log10(lambda_const)
            )
        # Take ny from reference file
    else:
        args["ny"] = (args["forcing"] / (lambda_const ** (8 / 3 * args["ny_n"]))) ** (
            1 / 2
        )

    print("args", args)

    if args["eigen_perturb"]:
        print("\nRunning with eigen_perturb\n")
        perturb_e_vectors, _, _ = find_eigenvector_for_perturbation(
            u_init_profiles[
                :,
                0 : args["n_profiles"]
                * args["n_runs_per_profile"] : args["n_runs_per_profile"],
            ],
            dev_plot_active=False,
            n_profiles=args["n_profiles"],
            local_ny=header_dict["ny"],
        )
    else:
        print("\nRunning without eigen_perturb\n")
        perturb_e_vectors = np.ones((n_k_vec, args["n_profiles"]), dtype=np.complex128)

    if args["single_shell_perturb"] is not None:
        print("\nRunning in single shell perturb mode\n")

    # Make perturbations
    perturbations = calculate_perturbations(
        perturb_e_vectors, dev_plot_active=False, args=args
    )

    # Prepare array for saving
    data_out = np.zeros(
        (int(args["Nt"] * sample_rate) + args["endpoint"] * 1, n_k_vec + 1),
        dtype=np.complex128,
    )

    # Detect if other perturbations exist in the perturbation_folder and calculate
    # perturbation count to start at
    # Check if path exists
    expected_path = Path(args["path"], args["perturb_folder"])
    dir_exists = os.path.isdir(expected_path)
    if dir_exists:
        n_perturbation_files = len(list(expected_path.glob("*.csv")))
    else:
        n_perturbation_files = 0

    # Prepare and start the perturbation_runner in multiple processes
    processes = []
    profiler.start()

    # Get number of threads
    cpu_count = multiprocessing.cpu_count()

    # Append processes
    for j in range(args["n_runs_per_profile"] * args["n_profiles"] // cpu_count):
        for i in range(cpu_count):
            count = j * cpu_count + i

            processes.append(
                multiprocessing.Process(
                    target=perturbation_runner,
                    args=(
                        u_init_profiles[:, count] + perturbations[:, count],
                        perturb_positions,
                        du_array,
                        data_out,
                        args,
                        count,
                        count + n_perturbation_files,
                    ),
                )
            )
            processes[-1].start()

        for i in range(len(processes)):
            processes[i].join()

        processes = []

    for i in range(args["n_runs_per_profile"] * args["n_profiles"] % cpu_count):
        count = (
            args["n_runs_per_profile"] * args["n_profiles"] // cpu_count
        ) * cpu_count + i

        processes.append(
            multiprocessing.Process(
                target=perturbation_runner,
                args=(
                    u_init_profiles[:, count] + perturbations[:, count],
                    perturb_positions,
                    du_array,
                    data_out,
                    args,
                    count,
                    count + n_perturbation_files,
                ),
            )
        )
        processes[-1].start()

    for i in range(len(processes)):
        processes[i].join()

    profiler.stop()
    print(profiler.output_text())

    save_perturb_info(args=args)


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", nargs="+", type=str)
    arg_parser.add_argument("--path", nargs="?", type=str)
    arg_parser.add_argument(
        "--perturb_folder", nargs="?", default=None, required=True, type=str
    )
    arg_parser.add_argument("--time_to_run", default=0.1, type=float)
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--ny_n", default=None, type=int)
    arg_parser.add_argument("--n_runs_per_profile", default=1, type=int)
    arg_parser.add_argument("--n_profiles", default=1, type=int)
    arg_parser.add_argument("--start_time", nargs="+", type=float)
    arg_parser.add_argument("--eigen_perturb", action="store_true")
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    arg_parser.add_argument("--single_shell_perturb", default=None, type=int)
    arg_parser.add_argument("--start_time_offset", default=None, type=float)
    arg_parser.add_argument("--endpoint", action="store_true")

    args = vars(arg_parser.parse_args())

    args["ref_run"] = False

    if args["start_time"] is not None:
        if args["n_profiles"] > 1 and args["start_time_offset"] is None:
            np.testing.assert_equal(
                len(args["start_time"]),
                args["n_profiles"],
                "The number of start times do not equal the number of"
                + " requested profiles.",
            )
        elif args["n_profiles"] > 1 and args["start_time_offset"] is not None:
            np.testing.assert_equal(
                len(args["start_time"]), 1, "Too many start times given"
            )
            print(
                "Determining starttimes from single starttime value and the"
                + " start_time_offset parameter"
            )
            args["start_time"] = [
                args["start_time"][0] + args["start_time_offset"] * i
                for i in range(args["n_profiles"])
            ]
        else:
            np.testing.assert_equal(
                len(args["start_time"]), 1, "Too many start times given"
            )

    # Set seed if wished
    if args["seed_mode"]:
        np.random.seed(seed=1)

    main(args=args)
