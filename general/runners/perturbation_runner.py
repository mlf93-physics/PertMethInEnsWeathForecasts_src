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

or

python ../general/runners/perturbation_runner.py
--out_exp_folder=test2
--time_to_run=0.1
--n_runs_per_profile=4
--pert_mode=bv_eof
--pert_vector_folder=pt_vectors
--exp_folder=shell_model_breed_vectors_4cycles

"""

import os
import sys

sys.path.append("..")
import copy
import multiprocessing
import pathlib as pl

import config as cfg
import general.analyses.breed_vector_eof_analysis as bv_eof_anal
import general.utils.argument_parsers as a_parsers
import general.utils.exceptions as g_exceptions
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.perturb_utils as pt_utils
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_perturbation as pt_save
import general.utils.user_interface as g_ui
import general.utils.runner_utils as r_utils
import general.utils.util_funcs as g_utils
import numpy as np
from general.params.experiment_licences import Experiments as EXP
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from pyinstrument import Profiler

if cfg.MODEL == Models.SHELL_MODEL:
    # Shell model specific imports
    import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
    import shell_model_experiments.utils.special_params as sh_sparams
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR
    from shell_model_experiments.params.params import ParamsStructType
    from shell_model_experiments.sabra_model.tl_sabra_model import (
        run_model as sh_tl_model,
    )
    from shell_model_experiments.sabra_model.sabra_model import run_model as sh_model

    # Get parameters for model
    params = PAR
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    # Lorentz-63 model specific imports
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.utils.util_funcs as ut_funcs
    from lorentz63_experiments.lorentz63_model.lorentz63 import run_model as l63_model
    from lorentz63_experiments.lorentz63_model.tl_lorentz63 import (
        run_model as l63_tl_model,
    )
    from lorentz63_experiments.lorentz63_model.atl_lorentz63 import (
        run_model as l63_atl_model,
    )

    # Get parameters for model
    params = l63_params
    sparams = l63_sparams

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def perturbation_runner(
    u_old: np.ndarray,
    perturb_positions: List[int],
    du_array: np.ndarray,
    data_out_list: List[np.ndarray],
    args: dict,
    run_count: int,
    perturb_count: int,
    u_ref: Union[np.ndarray, None] = None,
):
    """Execute a given model on one given perturbed u_old profile

    Parameters
    ----------
    u_old : np.ndarray
        The previous velocity vector (i.e. the initial vel. vector)
    perturb_positions : List[int]
        List of perturbation index positions
    du_array : np.ndarray
        Array to store the du vector
    data_out_list : List[np.ndarray]
        List to store output data from processes (i.e. when LICENCE = BREEDING_VECTORS)
    args : dict
        Run-time arguments
    run_count : int
        Counter that holds the run number of the current process
    perturb_count : int
        Counter that holds the perturbation number of the current process
        (i.e. including the number of existing perturbations)
    u_ref : Union[np.ndarray, None], optional
        The reference data (needed for TL models), by default None
    """
    # Prepare array for saving
    data_out = np.zeros(
        (int(args["Nt"] * params.sample_rate) + args["endpoint"] * 1, params.sdim + 1),
        dtype=sparams.dtype,
    )

    print(
        f"Running perturbation {run_count + 1}/"
        + f"{args['n_profiles']*args['n_runs_per_profile']} | profile"
        + f" {run_count // args['n_runs_per_profile']}, profile run"
        + f" {run_count % args['n_runs_per_profile']}"
    )
    if cfg.MODEL == Models.SHELL_MODEL:
        # Get diffusion functions
        if args["diff_type"] == "inf_hyper":
            diff_function = ut_funcs.infinit_hyper_diffusion
        else:
            diff_function = ut_funcs.normal_diffusion

        if cfg.MODEL.submodel is None:
            sh_model(
                diff_function,
                u_old,
                data_out,
                args["Nt"] + args["endpoint"] * 1,
                args["ny"],
                args["forcing"],
                args["diff_exponent"],
                params,
            )
        elif cfg.MODEL.submodel == "TL":
            sh_tl_model(
                u_old,
                np.copy(u_ref[:, run_count]),
                data_out,
                args["Nt"] + args["endpoint"] * 1,
                args["ny"],
                args["diff_exponent"],
                args["forcing"],
                params,
            )
        else:
            g_exceptions.ModelError(
                "Submodel invalid or not implemented yet",
                model=f"{cfg.MODEL.submodel} {cfg.MODEL}",
            )
    elif cfg.MODEL == Models.LORENTZ63:
        if cfg.MODEL.submodel is None:
            # Model specific setup
            lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)
            l63_model(
                u_old,
                du_array,
                lorentz_matrix,
                data_out,
                args["Nt"] + args["endpoint"] * 1,
            )
        elif cfg.MODEL.submodel == "TL":
            jacobian_matrix = l63_nm_estimator.init_jacobian(args)
            lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)

            l63_tl_model(
                u_old,
                du_array,
                np.copy(u_ref[:, run_count]),
                lorentz_matrix,
                jacobian_matrix,
                data_out,
                args["Nt"] + args["endpoint"] * 1,
                r_const=args["r_const"],
                raw_perturbation=True,
            )
        elif cfg.MODEL.submodel == "ATL":
            jacobian_matrix = l63_nm_estimator.init_jacobian(args)
            lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)
            ref_data = np.zeros(
                (args["Nt"] + args["endpoint"] * 1, params.sdim), dtype=np.float64
            )

            l63_atl_model(
                np.reshape(u_old, (params.sdim, 1)),
                du_array,
                np.copy(u_ref[:, run_count]),
                lorentz_matrix,
                jacobian_matrix,
                data_out,
                ref_data,
                args["Nt"] + args["endpoint"] * 1,
                r_const=args["r_const"],
                raw_perturbation=True,
            )
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

    if (
        cfg.LICENCE == EXP.BREEDING_VECTORS
        or cfg.LICENCE == EXP.LYAPUNOV_VECTORS
        or cfg.LICENCE == EXP.SINGULAR_VECTORS
    ):
        # Save latest state vector to output dir
        data_out_list.append(data_out[-1, 1:])


def prepare_run_times(
    args: dict,
) -> Tuple[np.ndarray, np.ndarray]:

    num_perturbations = args["n_runs_per_profile"] * args["n_profiles"]

    if cfg.LICENCE != EXP.LORENTZ_BLOCK:
        times_to_run = np.ones(num_perturbations) * args["time_to_run"]
        Nt_array = np.round(times_to_run / params.dt).astype(np.int64)
    else:
        if len(args["start_times"]) > 1:
            start_times = np.array(args["start_times"])
        else:
            start_times = np.ones(num_perturbations) * args["start_times"]

        times_to_run = start_times[0] + args["time_to_run"] - start_times
        Nt_array = (times_to_run / params.dt).astype(np.int64)

    return times_to_run, Nt_array


def prepare_perturbations(
    args: dict, raw_perturbations: bool = False
) -> Tuple[np.ndarray, List[int]]:
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
        (
            u_return : perturbed u profiles
            perturb_positions : position of the perturbation (in samples)
        )


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

    # Only import start profiles beforehand if not using bv, bv_eof or sv perturbations,
    # i.e. also when running in singel_shell_perturb mode

    if args["pert_mode"] not in ["bv", "bv_eof", "sv"]:
        (
            u_init_profiles,
            perturb_positions,
            header_dict,
        ) = g_import.import_start_u_profiles(args=args)

    # NM pert generation mode; if True, the perturbations are generated in the
    # plane of the complex-conjugate pair of the leading NM. Otherwise only one
    # perturbation is made
    # nm_complex_conj = False

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
                (
                    perturb_vectors,
                    e_values_max,
                    _,
                    _,
                ) = l63_nm_estimator.find_normal_modes(
                    u_init_profiles[
                        :,
                        0 : args["n_profiles"]
                        * args["n_runs_per_profile"] : args["n_runs_per_profile"],
                    ],
                    args,
                    n_profiles=args["n_profiles"],
                )
        elif "bv" in args["pert_mode"]:
            (
                perturb_vectors,
                u_init_profiles,
                perturb_positions,
                _,
            ) = pt_import.import_perturb_vectors(args)

            if args["pert_mode"] == "bv_eof":
                print("\nRunning with BREED VECTOR EOF perturbations\n")

                eof_vectors: np.ndarray = bv_eof_anal.calc_bv_eof_vectors(
                    perturb_vectors, args["n_runs_per_profile"]
                )
                # Reshape and save as perturb_vectors
                perturb_vectors = np.reshape(
                    np.transpose(eof_vectors, axes=(1, 0, 2)),
                    (params.sdim, args["n_profiles"] * args["n_runs_per_profile"]),
                )
            else:
                print("\nRunning with BREED VECTOR perturbations\n")

        elif "sv" in args["pert_mode"]:
            print("\nRunning with SINGULAR VECTOR perturbations\n")
            (
                perturb_vectors,
                u_init_profiles,
                perturb_positions,
                _,
            ) = pt_import.import_perturb_vectors(args, raw_perturbations=True)
            # Reshape perturb_vectors
            perturb_vectors = np.reshape(
                np.transpose(perturb_vectors, axes=(2, 0, 1)),
                (params.sdim, args["n_profiles"] * args["n_runs_per_profile"]),
            )

        elif args["pert_mode"] == "rd":
            print("\nRunning with RANDOM perturbations\n")
            perturb_vectors = np.ones(
                (params.sdim, args["n_profiles"]), dtype=sparams.dtype
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
    u_profiles_perturbed: np.ndarray,
    perturb_positions: List[int],
    times_to_run: np.ndarray,
    Nt_array: np.ndarray,
    n_perturbation_files: int,
    u_ref: np.ndarray = None,
    args: dict = None,
) -> Tuple[List[multiprocessing.Process], List[np.ndarray]]:
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
    # Initiate arrays
    # params.initiate_sdim_arrays(args["sdim"])

    times_to_run, Nt_array = prepare_run_times(args)

    # If in 2. or higher breed cycle, the perturbation is given as input
    if u_profiles_perturbed is None or perturb_positions is None:

        raw_perturbations = False
        # Get only raw_perturbations if licence is LYAPUNOV_VECTORS
        if cfg.LICENCE == EXP.LYAPUNOV_VECTORS or cfg.LICENCE == EXP.SINGULAR_VECTORS:
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

    return processes, data_out_list, perturb_positions, u_profiles_perturbed


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

    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params, sdim=int(args["sdim"]))
        sh_utils.update_arrays(params)
    # Initiate arrays
    # params.initiate_sdim_arrays(args["sdim"])
    args = g_utils.adjust_start_times_with_offset(args)

    g_ui.confirm_run_setup(args)
    r_utils.adjust_run_setup(args)

    # Make profiler
    profiler = Profiler()
    # Start profiler
    profiler.start()

    processes, _, _, _ = main_setup(args=args)
    main_run(processes, args=args)

    profiler.stop()
    print(profiler.output_text(color=True))
