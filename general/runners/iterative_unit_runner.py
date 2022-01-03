import sys

sys.path.append("..")
import copy
import pathlib as pl
import numpy as np
import multiprocessing
import config as cfg
from general.params.model_licences import Models
from general.params.experiment_licences import Experiments as EXP
from general.utils.module_import.type_import import *
import general.utils.running.sv_runner_utils as sv_r_utils
import general.utils.running.runner_utils as r_utils
import general.utils.saving.save_data_funcs as g_save
from libs.libutils import file_utils as lib_file_utils


if cfg.MODEL == Models.SHELL_MODEL:
    # Shell model specific imports
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR
    from shell_model_experiments.params.params import ParamsStructType

    # Get parameters for model
    params = PAR
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    # Lorentz-63 model specific imports
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    # Get parameters for model
    params = l63_params
    sparams = l63_sparams


def unit_runner(
    u_old: np.ndarray,
    run_count: int,
    exp_setup: dict,
    args: dict,
    data_out_dict: dict,
    perturb_positions: np.ndarray,
    u_ref: Union[np.ndarray, None] = None,
):

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

    if cfg.LICENCE == EXP.SINGULAR_VECTORS:
        sv_matrix, s_values = sv_r_utils.sv_generator(
            exp_setup, args, u_ref, u_old, run_count, data_out
        )
        data_out_dict[run_count] = {"sv_matrix": sv_matrix, "s_values": s_values}


def prepare_processes(
    u_profiles_perturbed: np.ndarray,
    u_ref: np.ndarray,
    n_units: int,
    n_perturbation_files: int,
    perturb_positions: List[int],
    times_to_run: np.ndarray,
    Nt_array: np.ndarray,
    args: dict,
):
    # Prepare return data list
    manager = multiprocessing.Manager()
    data_out_dict = manager.dict()

    processes = []
    cpu_count = multiprocessing.cpu_count()

    # Append processes
    for j in range(n_units // cpu_count):
        for i in range(cpu_count):
            count = j * cpu_count + i

            args["time_to_run"] = times_to_run[count]
            args["Nt"] = Nt_array[count]

            # Copy args in order to avoid override between processes
            copy_args = copy.deepcopy(args)

            processes.append(
                multiprocessing.Process(
                    target=unit_runner,
                    args=(
                        np.copy(u_profiles_perturbed[:, count]),
                        copy_args,
                        count,
                        count + n_perturbation_files,
                        data_out_dict,
                        perturb_positions,
                    ),
                    kwargs={"u_ref": u_ref},
                )
            )

    for i in range(n_units % cpu_count):
        count = (n_units // cpu_count) * cpu_count + i

        args["time_to_run"] = times_to_run[count]
        args["Nt"] = Nt_array[count]

        # Copy args in order to avoid override between processes
        copy_args = copy.deepcopy(args)

        processes.append(
            multiprocessing.Process(
                target=unit_runner,
                args=(
                    np.copy(u_profiles_perturbed[:, count]),
                    copy_args,
                    count,
                    count + n_perturbation_files,
                    data_out_dict,
                    perturb_positions,
                ),
                kwargs={"u_ref": u_ref},
            )
        )

    return processes, data_out_dict


def main_setup(
    args: dict = None,
    exp_setup: dict = None,
    u_ref: np.ndarray = None,
    n_existing_files: int = 0,
) -> Tuple[List[multiprocessing.Process], dict]:

    times_to_run, Nt_array = r_utils.prepare_run_times(args)

    if cfg.LICENCE == EXP.SINGULAR_VECTORS:
        raw_perturbations = True

        u_profiles_perturbed, perturb_positions = r_utils.prepare_perturbations(
            args, raw_perturbations=raw_perturbations
        )

    # Detect if other perturbations exist in the perturbation_folder and calculate
    # perturbation count to start at
    expected_path = pl.Path(args["datapath"], args["out_exp_folder"])

    processes, data_out_dict = prepare_processes(
        u_profiles_perturbed,
        u_ref,
        n_existing_files,
        perturb_positions,
        times_to_run,
        Nt_array,
        args,
    )

    # if not args["skip_save_data"]:
    #     g_save.save_perturb_info(args=args, exp_setup=exp_setup)

    return processes, data_out_dict
