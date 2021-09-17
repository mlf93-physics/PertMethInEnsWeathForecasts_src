import os
import sys

sys.path.append("..")
import math
import copy
import argparse
from pathlib import Path
import numpy as np
import multiprocessing
from pyinstrument import Profiler
from shell_model_experiments.sabra_model.sabra_model import run_model as sh_model
import shell_model_experiments.params as sh_params
from shell_model_experiments.utils.save_data_funcs import save_lorentz_block_data
import shell_model_experiments.lyaponov.lyaponov_exp_estimator as lp_estimator
from lorentz63_experiments.lorentz63_model.lorentz63 import run_model as l63_model
import lorentz63_experiments.params as l63_params
import lorentz63_experiments.utils.util_funcs as ut_funcs
import general.utils.util_funcs as g_utils
import general.utils.import_data_funcs as g_import
from general.params.experiment_licences import Experiments as EXP
import general.utils.save_data_funcs as g_save
from general.params.model_licences import Models
from config import MODEL, LICENCE

profiler = Profiler()

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def perturbation_runner(
    u_old, perturb_positions, du_array, args, run_count, perturb_count
):
    """Execute the sabra model on one given perturbed u_old profile"""
    # Prepare array for saving
    data_out = np.zeros(
        (int(args["Nt"] * params.sample_rate) + args["endpoint"] * 1, params.sdim + 1),
        dtype=params.dtype,
    )

    print(
        f"Running perturbation {run_count + 1}/"
        + f"{args['n_profiles']*args['n_runs_per_profile']} | profile"
        + f" {run_count // args['n_runs_per_profile']}, profile run"
        + f" {run_count % args['n_runs_per_profile']}"
    )
    if MODEL == Models.SHELL_MODEL:
        sh_model(
            u_old,
            du_array,
            data_out,
            args["Nt"] + args["endpoint"] * 1,
            args["ny"],
            args["forcing"],
        )
    elif MODEL == Models.LORENTZ63:
        # Model specific setup
        derivMatrix = ut_funcs.setup_deriv_matrix(args)
        l63_model(
            u_old, du_array, derivMatrix, data_out, args["Nt"] + args["endpoint"] * 1
        )

    if LICENCE == EXP.NORMAL_PERTURBATION:
        g_save.save_data(
            data_out,
            prefix=f"perturb{perturb_count}_",
            perturb_position=perturb_positions[run_count // args["n_runs_per_profile"]],
            args=args,
        )
    elif LICENCE == EXP.LORENTZ_BLOCK:
        save_lorentz_block_data(
            data_out,
            prefix=f"lorentz{perturb_count}_",
            perturb_position=perturb_positions[run_count // args["n_runs_per_profile"]],
            args=args,
        )


def prepare_run_times(args):

    if LICENCE == EXP.NORMAL_PERTURBATION:
        num_perturbations = args["n_runs_per_profile"] * args["n_profiles"]
        times_to_run = np.ones(num_perturbations) * args["time_to_run"]
        Nt_array = (times_to_run / params.dt).astype(np.int64)

    elif LICENCE == EXP.LORENTZ_BLOCK:
        num_perturbations = args["n_runs_per_profile"] * args["n_profiles"]

        if len(args["start_time"]) > 1:
            start_times = np.array(args["start_time"])
        else:
            start_times = np.ones(num_perturbations) * args["start_time"]

        times_to_run = start_times[0] + args["time_to_run"] - start_times
        Nt_array = (times_to_run / params.dt).astype(np.int64)

    return times_to_run, Nt_array


def prepare_perturbations(args):

    # Import start profiles
    u_init_profiles, perturb_positions, header_dict = g_import.import_start_u_profiles(
        args=args
    )

    if MODEL == Models.SHELL_MODEL:
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
                    * math.log10(args["forcing"] / (header_dict["ny"] ** 2))
                    / math.log10(params.lambda_const)
                )
            # Take ny from reference file
        else:
            args["ny"] = (
                args["forcing"] / (params.lambda_const ** (8 / 3 * args["ny_n"]))
            ) ** (1 / 2)

    elif MODEL == Models.LORENTZ63:
        print("Nothing specific to do with args in lorentz63 model yet")

    if args["eigen_perturb"]:
        print("\nRunning with eigen_perturb\n")
        if MODEL == Models.SHELL_MODEL:
            perturb_e_vectors, _, _ = lp_estimator.find_eigenvector_for_perturbation(
                u_init_profiles[
                    :,
                    0 : args["n_profiles"]
                    * args["n_runs_per_profile"] : args["n_runs_per_profile"],
                ],
                dev_plot_active=False,
                n_profiles=args["n_profiles"],
                local_ny=header_dict["ny"],
            )
        elif MODEL == Models.LORENTZ63:
            print(
                "No eigenvector perturbations implemented for the lorentz63 model yet"
            )
    else:
        print("\nRunning without eigen_perturb\n")
        perturb_e_vectors = np.ones(
            (params.sdim, args["n_profiles"]), dtype=params.dtype
        )

    # Specific to shell model setup
    if MODEL == Models.SHELL_MODEL:
        if args["single_shell_perturb"] is not None:
            print("\nRunning in single shell perturb mode\n")

    # Make perturbations
    perturbations = lp_estimator.calculate_perturbations(
        perturb_e_vectors, dev_plot_active=False, args=args
    )

    return u_init_profiles, perturbations, perturb_positions


def prepare_processes(
    u_init_profiles,
    perturbations,
    perturb_positions,
    times_to_run,
    Nt_array,
    n_perturbation_files,
    args=None,
):

    # Prepare and start the perturbation_runner in multiple processes
    processes = []

    # Get number of threads
    cpu_count = multiprocessing.cpu_count()

    # Append processes
    for j in range(args["n_runs_per_profile"] * args["n_profiles"] // cpu_count):
        for i in range(cpu_count):
            count = j * cpu_count + i

            args["time_to_run"] = times_to_run[count]
            args["Nt"] = Nt_array[count]

            # Copy args in order to avoid override between processes
            copy_args = copy.deepcopy(args)

            processes.append(
                multiprocessing.Process(
                    target=perturbation_runner,
                    args=(
                        u_init_profiles[:, count] + perturbations[:, count],
                        perturb_positions,
                        params.du_array,
                        copy_args,
                        count,
                        count + n_perturbation_files,
                    ),
                )
            )

    for i in range(args["n_runs_per_profile"] * args["n_profiles"] % cpu_count):
        count = (
            args["n_runs_per_profile"] * args["n_profiles"] // cpu_count
        ) * cpu_count + i

        args["time_to_run"] = times_to_run[count]
        args["Nt"] = Nt_array[count]

        # Copy args in order to avoid override between processes
        copy_args = copy.deepcopy(args)

        processes.append(
            multiprocessing.Process(
                target=perturbation_runner,
                args=(
                    u_init_profiles[:, count] + perturbations[:, count],
                    perturb_positions,
                    params.du_array,
                    copy_args,
                    count,
                    count + n_perturbation_files,
                ),
            )
        )

    return processes


def main_setup(args=None):

    times_to_run, Nt_array = prepare_run_times(args)
    print("times_to_run, Nt_array", times_to_run, Nt_array)
    args["burn_in_lines"] = int(args["burn_in_time"] * params.tts)

    u_init_profiles, perturbations, perturb_positions = prepare_perturbations(args)

    # Detect if other perturbations exist in the perturbation_folder and calculate
    # perturbation count to start at
    # Check if path exists
    expected_path = Path(args["path"], args["perturb_folder"])
    dir_exists = os.path.isdir(expected_path)
    if dir_exists:
        n_perturbation_files = len(list(expected_path.glob("*.csv")))
    else:
        n_perturbation_files = 0

    processes = prepare_processes(
        u_init_profiles,
        perturbations,
        perturb_positions,
        times_to_run,
        Nt_array,
        n_perturbation_files,
        args=args,
    )

    g_save.save_perturb_info(args=args)

    return processes


def main_run(processes, args=None, num_blocks=None):

    cpu_count = multiprocessing.cpu_count()
    num_processes = len(processes)

    if num_blocks is not None:
        num_processes_per_block = 2 * args["n_profiles"] * args["n_runs_per_profile"]

    profiler.start()

    for j in range(num_processes // cpu_count):
        if num_blocks is not None:
            print(
                f"Block {int(j*cpu_count // num_processes_per_block)}-"
                + f"{int(((j + 1)*cpu_count // num_processes_per_block))}"
            )

        for i in range(cpu_count):
            count = j * cpu_count + i
            processes[count].start()

        for i in range(cpu_count):
            count = j * cpu_count + i
            processes[count].join()

    if num_blocks is not None:
        _dummy_done_count = (j + 1) * cpu_count // num_processes_per_block
        _dummy_remain_count = math.ceil(
            num_processes % cpu_count / num_processes_per_block
        )
        print(
            f"Block {int(_dummy_done_count)}-"
            + f"{int(_dummy_done_count + _dummy_remain_count)}"
        )
    for i in range(num_processes % cpu_count):

        count = (num_processes // cpu_count) * cpu_count + i
        processes[count].start()

    for i in range(num_processes % cpu_count):
        count = (num_processes // cpu_count) * cpu_count + i
        processes[count].join()

    profiler.stop()
    print(profiler.output_text())


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", nargs="+", type=str)
    arg_parser.add_argument("--path", nargs="?", type=str, required=True)
    arg_parser.add_argument(
        "--perturb_folder", nargs="?", default=None, required=True, type=str
    )
    arg_parser.add_argument("--time_to_run", default=0.1, type=float)
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--ny_n", default=None, type=int)
    arg_parser.add_argument("--n_runs_per_profile", default=1, type=int)
    arg_parser.add_argument("--n_profiles", default=1, type=int)
    arg_parser.add_argument("--start_time", nargs="+", type=float, required=True)
    arg_parser.add_argument("--eigen_perturb", action="store_true")
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    arg_parser.add_argument("--single_shell_perturb", default=None, type=int)
    arg_parser.add_argument("--start_time_offset", default=None, type=float)
    arg_parser.add_argument("--endpoint", action="store_true")

    args = vars(arg_parser.parse_args())

    args["ref_run"] = False

    args = g_utils.adjust_start_times_with_offset(args)

    # Set seed if wished
    if args["seed_mode"]:
        np.random.seed(seed=1)

    processes = main_setup(args=args)
    main_run(processes, args=args)
