import os
import sys

sys.path.append("..")
import math
import copy
import argparse
import pathlib as pl
import numpy as np
import multiprocessing
from pyinstrument import Profiler
from shell_model_experiments.sabra_model.sabra_model import run_model as sh_model
import shell_model_experiments.params as sh_params
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
from lorentz63_experiments.lorentz63_model.lorentz63 import run_model as l63_model
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.utils.util_funcs as ut_funcs
import general.utils.util_funcs as g_utils
import general.utils.importing.import_data_funcs as g_import
from general.params.experiment_licences import Experiments as EXP
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_perturbation as pt_save
import general.utils.perturb_utils as pt_utils
import general.utils.exceptions as g_exceptions
import general.utils.argument_parsers as a_parsers
from general.params.model_licences import Models
from config import MODEL, LICENCE, GLOBAL_PARAMS

profiler = Profiler()

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params

# Set global params
GLOBAL_PARAMS.ref_run = False


def perturbation_runner(
    u_old, perturb_positions, du_array, data_out_list, args, run_count, perturb_count
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
        deriv_matrix = ut_funcs.setup_deriv_matrix(args)
        l63_model(
            u_old, du_array, deriv_matrix, data_out, args["Nt"] + args["endpoint"] * 1
        )
    pt_save.save_perturbation_data(
        data_out,
        perturb_position=perturb_positions[run_count // args["n_runs_per_profile"]],
        perturb_count=perturb_count,
        args=args,
    )

    if LICENCE == EXP.BREEDING_VECTORS:
        # Save latest state vector to output dir
        data_out_list.append(data_out[-1, 1:])


def prepare_run_times(args):

    if LICENCE == EXP.NORMAL_PERTURBATION or LICENCE == EXP.BREEDING_VECTORS:
        num_perturbations = args["n_runs_per_profile"] * args["n_profiles"]
        times_to_run = np.ones(num_perturbations) * args["time_to_run"]
        Nt_array = (times_to_run / params.dt).astype(np.int64)

    elif LICENCE == EXP.LORENTZ_BLOCK:
        num_perturbations = args["n_runs_per_profile"] * args["n_profiles"]

        if len(args["start_times"]) > 1:
            start_times = np.array(args["start_times"])
        else:
            start_times = np.ones(num_perturbations) * args["start_times"]

        times_to_run = start_times[0] + args["time_to_run"] - start_times
        Nt_array = (times_to_run / params.dt).astype(np.int64)

    return times_to_run, Nt_array


def prepare_perturbations(args):

    # Import reference info file
    ref_header_dict = g_import.import_info_file(pl.Path(args["datapath"], "ref_data"))
    # header_dict = g_utils.handle_different_headers(header_dict)

    # Adjust parameters to have the correct ny/ny_n for shell model
    g_utils.determine_params_from_header_dict(ref_header_dict, args)

    # Only import start profiles beforehand if not using breed_vector perturbations,
    # i.e. also when running in singel_shell_perturb mode
    if args["pert_mode"] != "breed_vectors":
        (
            u_init_profiles,
            perturb_positions,
            header_dict,
        ) = g_import.import_start_u_profiles(args=args)

    if args["pert_mode"] is not None:

        if args["pert_mode"] == "nm":
            print("\nRunning with NORMAL MODE perturbations\n")
            if MODEL == Models.SHELL_MODEL:
                (perturb_vectors, _, _,) = sh_nm_estimator.find_normal_modes(
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
                perturb_vectors, _, _, _ = l63_nm_estimator.find_normal_modes(
                    u_init_profiles[
                        :,
                        0 : args["n_profiles"]
                        * args["n_runs_per_profile"] : args["n_runs_per_profile"],
                    ],
                    args,
                    dev_plot_active=False,
                    n_profiles=args["n_profiles"],
                )
        elif args["pert_mode"] == "bv":
            print("\nRunning with BREED VECTOR perturbations\n")
            (
                u_init_profiles,
                perturb_positions,
                perturb_vectors,
            ) = pt_utils.prepare_breed_vectors(args)

        elif args["pert_mode"] == "rd":
            print("\nRunning with RANDOM perturbations\n")
            perturb_vectors = np.ones(
                (params.sdim, args["n_profiles"]), dtype=params.dtype
            )
    # Check if single shell perturb should be activated
    elif args["single_shell_perturb"] is not None:
        # Specific to shell model setup
        if MODEL == Models.SHELL_MODEL:
            print("\nRunning in single shell perturb mode\n")
            perturb_vectors = None
        else:
            raise g_exceptions.ModelError(model=MODEL)
    else:
        _pert_arg = (
            "pert_mode: " + args["pert_mode"]
            if "pert_mode" in args
            else "single_shell_perturb: " + args["single_shell_perturb"]
        )
        raise g_exceptions.InvalidArgument(
            "Not a valid perturbation mode", argument=_pert_arg
        )

    # Make perturbations
    perturbations = pt_utils.calculate_perturbations(
        perturb_vectors, dev_plot_active=False, args=args
    )

    # Apply perturbations
    u_profiles_perturbed = u_init_profiles + perturbations
    return u_profiles_perturbed, perturb_positions


def prepare_processes(
    u_profiles_perturbed,
    perturb_positions,
    times_to_run,
    Nt_array,
    n_perturbation_files,
    args=None,
):
    # Prepare and start the perturbation_runner in multiple processes
    processes = []
    # Prepare return data list
    manager = multiprocessing.Manager()
    data_out_list = manager.list()
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
                        np.copy(u_profiles_perturbed[:, count]),
                        perturb_positions,
                        params.du_array,
                        data_out_list,
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
                    np.copy(u_profiles_perturbed[:, count]),
                    perturb_positions,
                    params.du_array,
                    data_out_list,
                    copy_args,
                    count,
                    count + n_perturbation_files,
                ),
            )
        )

    return processes, data_out_list


def main_setup(
    args=None, u_profiles_perturbed=None, perturb_positions=None, exp_setup=None
):

    times_to_run, Nt_array = prepare_run_times(args)

    if u_profiles_perturbed is None or perturb_positions is None:
        u_profiles_perturbed, perturb_positions = prepare_perturbations(args)

    # Detect if other perturbations exist in the perturbation_folder and calculate
    # perturbation count to start at
    # Check if path exists
    expected_path = pl.Path(args["datapath"], args["exp_folder"])
    dir_exists = os.path.isdir(expected_path)
    if dir_exists:
        n_perturbation_files = len(list(expected_path.glob("*.csv")))
    else:
        n_perturbation_files = 0

    processes, data_out_list = prepare_processes(
        u_profiles_perturbed,
        perturb_positions,
        times_to_run,
        Nt_array,
        n_perturbation_files,
        args=args,
    )

    g_save.save_perturb_info(args=args, exp_setup=exp_setup)

    return processes, data_out_list, perturb_positions


def main_run(processes, args=None, n_units=None):
    """Run the processes in parallel. The processes are distributed according
    to the number of units to run

    Parameters
    ----------
    processes : list
        List of processes to run
    args : dict, optional
        Run-time arguments, by default None
    n_units : int, optional
        The number of units to run (e.g. blocks, vectors etc. depending on
        the experiment license), by default None
    """

    cpu_count = multiprocessing.cpu_count()
    num_processes = len(processes)

    # if n_units is not None:
    #     if LICENCE == EXP.NORMAL_PERTURBATION or LICENCE == EXP.BREEDING_VECTORS:
    #         num_processes_per_unit = args["n_profiles"] * args["n_runs_per_profile"]
    #     elif LICENCE == EXP.LORENTZ_BLOCK:
    #         num_processes_per_unit = 2 * args["n_profiles"] * args["n_runs_per_profile"]

    profiler.start()

    for j in range(num_processes // cpu_count):
        # if n_units is not None:
        #     print(
        #         f"Unit {int(j*cpu_count // num_processes_per_unit)}-"
        #         + f"{int(((j + 1)*cpu_count // num_processes_per_unit))}"
        #     )

        for i in range(cpu_count):
            count = j * cpu_count + i
            processes[count].start()

        for i in range(cpu_count):
            count = j * cpu_count + i
            processes[count].join()
            processes[count].close()

    # if n_units is not None:
    #     _dummy_done_count = (j + 1) * cpu_count // num_processes_per_unit
    #     _dummy_remain_count = math.ceil(
    #         num_processes % cpu_count / num_processes_per_unit
    #     )
    #     print(
    #         f"Unit {int(_dummy_done_count)}-"
    #         + f"{int(_dummy_done_count + _dummy_remain_count)}"
    #     )
    for i in range(num_processes % cpu_count):

        count = (num_processes // cpu_count) * cpu_count + i
        processes[count].start()

    for i in range(num_processes % cpu_count):
        count = (num_processes // cpu_count) * cpu_count + i
        processes[count].join()
        processes[count].close()

    profiler.stop()
    print(profiler.output_text())


if __name__ == "__main__":
    # Get arguments
    pert_arg_setup = a_parsers.PerturbationArgSetup()
    pert_arg_setup.setup_parser()
    pert_arg_setup.validate_arguments()
    args = pert_arg_setup.args

    args = g_utils.adjust_start_times_with_offset(args)

    print("args", args)

    processes, _, _ = main_setup(args=args)
    main_run(processes, args=args)
