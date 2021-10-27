"""The perturbation runner.

Run a desired number of perturbation, given a desired perturbation method,
in parallel. The start times, number of runs, time to run etc. can be specified

Example to run:

python ../general/runners/perturbation_runner.py
--exp_folder=test1_perturbations
--time_to_run=0.1
--n_profiles=2
--start_times 4 8
--pert_mode=rd

"""

import os
import sys

sys.path.append("..")
import copy
import pathlib as pl
import numpy as np
import multiprocessing
from pyinstrument import Profiler
from shell_model_experiments.sabra_model.sabra_model import run_model as sh_model
import shell_model_experiments.params as sh_params
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
from lorentz63_experiments.lorentz63_model.lorentz63 import run_model as l63_model
from lorentz63_experiments.lorentz63_model.tl_lorentz63 import run_model as l63_tl_model
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.utils.util_funcs as ut_funcs
import general.utils.util_funcs as g_utils
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
from general.params.experiment_licences import Experiments as EXP
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_perturbation as pt_save
import general.analyses.breed_vector_eof_analysis as bv_eof_anal
import general.utils.perturb_utils as pt_utils
import general.utils.exceptions as g_exceptions
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
from general.params.model_licences import Models
import config as cfg


# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = sh_params
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def perturbation_runner(
    u_old,
    perturb_positions,
    du_array,
    data_out_list,
    args,
    run_count,
    perturb_count,
    u_ref=None,
):
    """Execute a given model on one given perturbed u_old profile"""
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
    if cfg.MODEL == Models.SHELL_MODEL:
        if cfg.MODEL.submodel is None:
            sh_model(
                u_old,
                du_array,
                data_out,
                args["Nt"] + args["endpoint"] * 1,
                args["ny"],
                args["forcing"],
                args["diff_exponent"],
            )
        else:
            g_exceptions.ModelError(
                "Submodel invalid or not implemented yet",
                model=f"{cfg.MODEL.submodel} {cfg.MODEL}",
            )
    elif cfg.MODEL == Models.LORENTZ63:
        if cfg.MODEL.submodel is None:
            # Model specific setup
            deriv_matrix = ut_funcs.setup_deriv_matrix(args)
            l63_model(
                u_old,
                du_array,
                deriv_matrix,
                data_out,
                args["Nt"] + args["endpoint"] * 1,
            )
        elif cfg.MODEL.submodel == "TL":
            deriv_matrix = l63_nm_estimator.init_jacobian(args)

            l63_tl_model(
                u_old,
                du_array,
                u_ref,
                deriv_matrix,
                data_out,
                args["Nt"] + args["endpoint"] * 1,
                r_const=args["r_const"],
            )

            # Add reference data to TL model trajectory, since only the perturbation
            # is integrated in the model
            data_out[:, 1:] += u_ref

        else:
            g_exceptions.ModelError(
                "Submodel invalid or not implemented yet",
                model=f"{cfg.MODEL.submodel} {cfg.MODEL}",
            )

    if not args["skip_save_data"]:
        pt_save.save_perturbation_data(
            data_out,
            perturb_position=perturb_positions[run_count // args["n_runs_per_profile"]],
            perturb_count=perturb_count,
            args=args,
        )

    if cfg.LICENCE == EXP.BREEDING_VECTORS or cfg.LICENCE == EXP.LYAPUNOV_VECTORS:
        # Save latest state vector to output dir
        data_out_list.append(data_out[-1, 1:])


def prepare_run_times(args):

    if cfg.LICENCE != EXP.LORENTZ_BLOCK:
        num_perturbations = args["n_runs_per_profile"] * args["n_profiles"]
        times_to_run = np.ones(num_perturbations) * args["time_to_run"]
        Nt_array = (times_to_run / params.dt).astype(np.int64)

    else:
        num_perturbations = args["n_runs_per_profile"] * args["n_profiles"]

        if len(args["start_times"]) > 1:
            start_times = np.array(args["start_times"])
        else:
            start_times = np.ones(num_perturbations) * args["start_times"]

        times_to_run = start_times[0] + args["time_to_run"] - start_times
        Nt_array = (times_to_run / params.dt).astype(np.int64)

    return times_to_run, Nt_array


def prepare_perturbations(args: dict, raw_perturbations: bool = False):
    """Prepare the perturbed initial conditions according to the desired perturbation
    type (given by args["pert_mode"])

    Parameters
    ----------
    args : dict
        Run-time arguments
    raw_perturbations : bool, optional
        If the raw perturbations should be returned instead of the perturbations
        added to the u_init_profiles, by default False

    Returns
    -------
    tuple
        (perturbed u profiles, position of the perturbation (in samples))

    Raises
    ------
    g_exceptions.ModelError
        Raised if the single_shell_perturb option is set while not using the shell
        model
    g_exceptions.InvalidRuntimeArgument
        Raised if the perturbation mode is not valid
    """

    # Import reference info file
    ref_header_dict = g_import.import_info_file(pl.Path(args["datapath"], "ref_data"))
    # header_dict = g_utils.handle_different_headers(header_dict)

    # Adjust parameters to have the correct ny/ny_n for shell model
    g_utils.determine_params_from_header_dict(ref_header_dict, args)

    # Only import start profiles beforehand if not using breed_vector perturbations,
    # i.e. also when running in singel_shell_perturb mode
    if args["pert_mode"] not in ["bv", "bv_eof"]:
        (
            u_init_profiles,
            perturb_positions,
            header_dict,
        ) = g_import.import_start_u_profiles(args=args)

    if args["pert_mode"] is not None:

        if args["pert_mode"] == "nm":
            print("\nRunning with NORMAL MODE perturbations\n")
            if cfg.MODEL == Models.SHELL_MODEL:
                (perturb_vectors, _, _,) = sh_nm_estimator.find_normal_modes(
                    u_init_profiles[
                        :,
                        0 : args["n_profiles"]
                        * args["n_runs_per_profile"] : args["n_runs_per_profile"],
                    ],
                    args,
                    dev_plot_active=False,
                    local_ny=header_dict["ny"],
                )
            elif cfg.MODEL == Models.LORENTZ63:
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
                perturb_vectors,
                u_init_profiles,
                perturb_positions,
                _,
            ) = pt_import.import_perturb_vectors(args)

            # Reshape perturb_vectors
            perturb_vectors = np.reshape(
                np.transpose(perturb_vectors, axes=(2, 0, 1)),
                (params.sdim, args["n_profiles"] * args["n_runs_per_profile"]),
            )

        elif args["pert_mode"] == "bv_eof":
            print("\nRunning with BREED VECTOR EOF perturbations\n")

            (
                breed_vectors,
                u_init_profiles,
                perturb_positions,
                _,
            ) = pt_import.import_perturb_vectors(args)

            eof_vectors: np.ndarray = bv_eof_anal.calc_bv_eof_vectors(
                breed_vectors, args["n_runs_per_profile"]
            )
            # Reshape and save as perturb_vectors
            perturb_vectors = np.reshape(
                np.transpose(eof_vectors, axes=(1, 0, 2)),
                (params.sdim, args["n_profiles"] * args["n_runs_per_profile"]),
            )

        elif args["pert_mode"] == "rd":
            print("\nRunning with RANDOM perturbations\n")
            perturb_vectors = np.ones(
                (params.sdim, args["n_profiles"]), dtype=params.dtype
            )
    # Check if single shell perturb should be activated
    elif args["single_shell_perturb"] is not None:
        # Specific to shell model setup
        if cfg.MODEL == Models.SHELL_MODEL:
            print("\nRunning in single shell perturb mode\n")
            perturb_vectors = None
        else:
            raise g_exceptions.ModelError(model=cfg.MODEL)
    else:
        _pert_arg = (
            "pert_mode: " + args["pert_mode"]
            if "pert_mode" in args
            else "single_shell_perturb: " + args["single_shell_perturb"]
        )
        raise g_exceptions.InvalidRuntimeArgument(
            "Not a valid perturbation mode", argument=_pert_arg
        )

    # Make perturbations
    perturbations = pt_utils.calculate_perturbations(
        perturb_vectors, dev_plot_active=False, args=args
    )

    if raw_perturbations:
        # Return raw perturbations
        u_return = perturbations
    else:
        # Apply perturbations
        u_return = u_init_profiles + perturbations

    return u_return, perturb_positions


def prepare_processes(
    u_profiles_perturbed,
    perturb_positions,
    times_to_run,
    Nt_array,
    n_perturbation_files,
    u_ref=None,
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
                    kwargs={"u_ref": u_ref},
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
                kwargs={"u_ref": u_ref},
            )
        )

    return processes, data_out_list


def main_setup(
    args=None,
    u_profiles_perturbed=None,
    perturb_positions=None,
    exp_setup=None,
    u_ref=None,
):

    times_to_run, Nt_array = prepare_run_times(args)

    # If in 2. or higher breed cycle, the perturbation is given as input
    if u_profiles_perturbed is None or perturb_positions is None:

        raw_perturbations = False
        # Get only raw_perturbations if licence is LYAPUNOV_VECTORS
        if cfg.LICENCE == EXP.LYAPUNOV_VECTORS:
            raw_perturbations = True

        u_profiles_perturbed, perturb_positions = prepare_perturbations(
            args, raw_perturbations=raw_perturbations
        )

    # Detect if other perturbations exist in the perturbation_folder and calculate
    # perturbation count to start at
    # Check if path exists
    expected_path = pl.Path(args["datapath"], args["out_exp_folder"])
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
        u_ref=u_ref,
        args=args,
    )

    if not args["skip_save_data"]:
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
    #     if cfg.LICENCE == EXP.NORMAL_PERTURBATION or cfg.LICENCE == EXP.BREEDING_VECTORS:
    #         num_processes_per_unit = args["n_profiles"] * args["n_runs_per_profile"]
    #     elif cfg.LICENCE == EXP.LORENTZ_BLOCK:
    #         num_processes_per_unit = 2 * args["n_profiles"] * args["n_runs_per_profile"]

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


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    pert_arg_setup = a_parsers.PerturbationArgSetup()
    pert_arg_setup.setup_parser()
    pert_arg_setup.validate_arguments()
    args = pert_arg_setup.args

    args = g_utils.adjust_start_times_with_offset(args)

    g_ui.confirm_run_setup(args)

    # Make profiler
    profiler = Profiler()
    # Start profiler
    profiler.start()

    processes, _, _ = main_setup(args=args)
    main_run(processes, args=args)

    profiler.stop()
    print(profiler.output_text())
