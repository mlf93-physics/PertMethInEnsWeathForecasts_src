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
import itertools as it
import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.exceptions as g_exceptions
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_perturbation as pt_save
import general.utils.user_interface as g_ui
import general.utils.running.runner_utils as r_utils
import general.utils.process_utils as pr_utils
import general.utils.util_funcs as g_utils
import numpy as np
from general.params.experiment_licences import Experiments as EXP
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from libs.libutils import file_utils as lib_file_utils
from pyinstrument import Profiler

if cfg.MODEL == Models.SHELL_MODEL:
    # Shell model specific imports
    import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
    import shell_model_experiments.utils.special_params as sh_sparams
    import shell_model_experiments.utils.util_funcs as ut_funcs
    from shell_model_experiments.params.params import PAR
    from shell_model_experiments.params.params import ParamsStructType
    import shell_model_experiments.utils.runner_utils as sh_r_utils
    from shell_model_experiments.sabra_model.tl_sabra_model import (
        run_model as sh_tl_model,
    )
    from shell_model_experiments.sabra_model.atl_sabra_model import (
        run_model as sh_atl_model,
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
        else:
            # Initialise the Jacobian and diagonal arrays
            (
                J_matrix,
                diagonal0,
                diagonal1,
                diagonal2,
                diagonal_1,
                diagonal_2,
            ) = sh_nm_estimator.init_jacobian()

            if cfg.MODEL.submodel == "TL":
                sh_tl_model(
                    u_old,
                    np.copy(u_ref[:, run_count]),
                    data_out,
                    args["Nt"] + args["endpoint"] * 1,
                    args["ny"],
                    args["diff_exponent"],
                    args["forcing"],
                    params,
                    J_matrix,
                    diagonal0,
                    diagonal1,
                    diagonal2,
                    diagonal_1,
                    diagonal_2,
                )
            elif cfg.MODEL.submodel == "ATL":
                sh_atl_model(
                    u_old,
                    np.copy(u_ref[:, run_count]),
                    data_out,
                    args["Nt"] + args["endpoint"] * 1,
                    args["ny"],
                    args["diff_exponent"],
                    args["forcing"],
                    params,
                    J_matrix,
                    diagonal0,
                    diagonal1,
                    diagonal2,
                    diagonal_1,
                    diagonal_2,
                )
            else:
                g_exceptions.ModelError(
                    "Submodel invalid or not implemented yet",
                    model=f"{cfg.MODEL.submodel} {cfg.MODEL}",
                )
    elif cfg.MODEL == Models.LORENTZ63:
        # General model setup
        lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)

        # Submodel specific setup
        if cfg.MODEL.submodel is None:
            l63_model(
                u_old,
                lorentz_matrix,
                data_out,
                args["Nt"] + args["endpoint"] * 1,
            )
        else:
            # Common to all submodels
            jacobian_matrix = l63_nm_estimator.init_jacobian(args)

            if cfg.MODEL.submodel == "TL":
                l63_tl_model(
                    u_old,
                    np.copy(u_ref[:, run_count]),
                    lorentz_matrix,
                    jacobian_matrix,
                    data_out,
                    args["Nt"] + args["endpoint"] * 1,
                    r_const=args["r_const"],
                    raw_perturbation=True,
                )
            elif cfg.MODEL.submodel == "ATL":
                l63_atl_model(
                    np.reshape(u_old, (params.sdim, 1)),
                    np.copy(u_ref[:, run_count]),
                    lorentz_matrix,
                    jacobian_matrix,
                    data_out,
                    args["Nt"] + args["endpoint"] * 1,
                    r_const=args["r_const"],
                    raw_perturbation=True,
                )
            else:
                g_exceptions.ModelError(
                    "Submodel invalid or not implemented yet",
                    model=f"{cfg.MODEL.submodel} {cfg.MODEL}",
                )

    if not args["skip_save_data"] and not args["save_no_pert"]:
        pt_save.save_perturbation_data(
            data_out,
            perturb_position=perturb_positions[run_count // args["n_runs_per_profile"]],
            perturb_count=perturb_count,
            run_count=run_count,
            args=args,
        )

    if (
        cfg.LICENCE == EXP.BREEDING_VECTORS
        or cfg.LICENCE == EXP.LYAPUNOV_VECTORS
        or cfg.LICENCE == EXP.SINGULAR_VECTORS
    ):
        # For the ATL model, time goes backwards, i.e. first datapoint in data_out
        # stores the result of the last integration.
        if cfg.MODEL.submodel == "ATL":
            data_out_list.append(data_out[0, 1:])
        else:
            # Save latest state vector to output dir
            data_out_list.append(data_out[-1, 1:])


def prepare_processes(
    u_profiles_perturbed: np.ndarray,
    perturb_positions: List[int],
    times_to_run: np.ndarray,
    Nt_array: np.ndarray,
    n_perturbation_files: int,
    u_ref: np.ndarray = None,
    exec_all_runs_per_profile: Union[np.ndarray, None] = None,
    args: dict = None,
) -> Tuple[List[multiprocessing.Process], List[np.ndarray]]:
    # Prepare and start the perturbation_runner in multiple processes
    processes = []
    # Prepare return data list
    manager = multiprocessing.Manager()
    data_out_list = manager.list()

    # Make iterator; enables us to skip a number of runs if needed (e.g. for NM
    # perturbations in L63 model)
    iterator = iter(range(args["n_runs_per_profile"] * args["n_profiles"]))
    # Append processes
    for count in iterator:
        profile_count = count // args["n_runs_per_profile"]

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
                    data_out_list,
                    copy_args,
                    count,
                    count + n_perturbation_files,
                ),
                kwargs={"u_ref": u_ref},
            )
        )

        # Skip remaining runs per profile if requested
        if exec_all_runs_per_profile is not None:
            if not exec_all_runs_per_profile[profile_count]:
                # NOTE: Reason for -2: -1, since one run has already been added; -1,
                # since next skips one on its own
                if args["n_runs_per_profile"] == 1:
                    continue
                else:
                    islice_start = args["n_runs_per_profile"] - 2
                    next(it.islice(iterator, islice_start, None))

    return processes, data_out_list


def main_setup(
    args=None,
    u_profiles_perturbed=None,
    perturb_positions=None,
    exp_setup=None,
    u_ref=None,
):

    times_to_run, Nt_array = r_utils.prepare_run_times(args)
    exec_all_runs_per_profile = None

    # If in 2. or higher breed cycle, the perturbation is given as input
    if u_profiles_perturbed is None:  # or perturb_positions is None:

        raw_perturbations = False
        # Get only raw_perturbations if licence is LYAPUNOV_VECTORS
        if cfg.LICENCE == EXP.LYAPUNOV_VECTORS or cfg.LICENCE == EXP.SINGULAR_VECTORS:
            raw_perturbations = True

        (
            u_profiles_perturbed,
            perturb_positions,
            exec_all_runs_per_profile,
        ) = r_utils.prepare_perturbations(args, raw_perturbations=raw_perturbations)

    # Detect if other perturbations exist in the perturbation_folder and calculate
    # perturbation count to start at
    expected_path = pl.Path(args["datapath"], args["out_exp_folder"])
    n_perturbation_files = lib_file_utils.count_existing_files_or_dirs(expected_path)

    processes, data_out_list = prepare_processes(
        u_profiles_perturbed,
        perturb_positions,
        times_to_run,
        Nt_array,
        n_perturbation_files,
        u_ref=u_ref,
        exec_all_runs_per_profile=exec_all_runs_per_profile,
        args=args,
    )

    if not args["skip_save_data"] and not args["save_no_pert"]:
        g_save.save_perturb_info(args=args, exp_setup=exp_setup)

    return processes, data_out_list, perturb_positions, u_profiles_perturbed


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    _parser = a_parsers.PerturbationArgSetup()
    _parser.setup_parser()
    _parser.validate_arguments()
    _parser = (
        a_parsers.ReferenceAnalysisArgParser()
    )  # Needed for RF perturbations to work
    _parser.setup_parser()
    args = _parser.args

    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        ut_funcs.update_dependent_params(params)
        ut_funcs.update_arrays(params)

        if args["regime_start"] is not None:
            start_times, _, _ = sh_r_utils.get_regime_start_times(args)
            args["start_times"] = start_times[: args["n_profiles"]]

    if args["regime_start"] is None:
        args = g_utils.adjust_start_times_with_offset(args)

    g_ui.confirm_run_setup(args)
    r_utils.adjust_run_setup(args)

    # Make profiler
    profiler = Profiler()
    # Start profiler
    profiler.start()

    processes, _, _, _ = main_setup(args=args)
    pr_utils.main_run(processes)

    profiler.stop()
    print(profiler.output_text(color=True))
