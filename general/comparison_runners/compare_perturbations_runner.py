"""Runs a complete set of perturbations based on several different perturbation
methods [rd, bv, bv-eof, ...]. Used to compare the different perturbation methods

Example
-------


"""

import sys

sys.path.append("..")
import pathlib as pl
import copy
import shell_model_experiments.params as sh_params
import shell_model_experiments.utils.util_funcs as sh_utils
import lorentz63_experiments.params.params as l63_params
import general.utils.experiments.exp_utils as exp_utils
import general.utils.user_interface as g_ui
import general.utils.argument_parsers as a_parsers
import general.utils.runner_utils as r_utils
from general.runners.breed_vector_runner import main as bv_runner
import general.runners.perturbation_runner as pt_runner
from general.params.experiment_licences import Experiments as exp
import general.utils.util_funcs as g_utils
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = sh_params
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def generate_bvs(args: dict, exp_setup: dict):
    """Generate the BVs according to the exp setup

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """
    # Update licence
    cfg.LICENCE = exp.BREEDING_VECTORS

    # Set local args params
    args["pert_mode"] = "rd"

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["bv_gen_setup"]
    local_exp_setup["folder_name"] += "/breed_vectors"

    bv_runner(args, local_exp_setup)


def generate_vectors(args: dict, exp_setup: dict):
    """Generate all perturbation vectors (i.e. BVs and SVs)

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """
    args["save_last_pert"] = True

    generate_bvs(copy.deepcopy(args), exp_setup)


def rd_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with rd as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = 50
    args["pert_mode"] = "rd"
    args["start_times"] = local_exp_setup["eval_times"]
    args["start_time_offset"] = local_exp_setup["unit_offset"]
    args["out_exp_folder"] = local_exp_setup["folder_name"] + "/rd_perturbations"

    # Adjust start times
    args = g_utils.adjust_start_times_with_offset(args)

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    temp_processes, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    r_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def nm_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with nm as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = 50
    args["n_profiles"] = local_exp_setup["n_units"]
    args["pert_mode"] = "nm"
    args["start_times"] = local_exp_setup["eval_times"]
    args["start_time_offset"] = local_exp_setup["unit_offset"]
    args["out_exp_folder"] = local_exp_setup["folder_name"] + "/nm_perturbations"

    # Adjust start times
    args = g_utils.adjust_start_times_with_offset(args)

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    temp_processes, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    r_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def bv_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with BVs as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = 4
    args["pert_mode"] = "bv"
    args["pert_vector_folder"] = local_exp_setup["folder_name"]
    args["exp_folder"] = "breed_vectors"
    args["out_exp_folder"] = local_exp_setup["folder_name"] + "/bv_perturbations"

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    temp_processes, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    r_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def bv_eof_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with BV-EOFs as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = 3
    args["pert_mode"] = "bv_eof"
    args["pert_vector_folder"] = local_exp_setup["folder_name"]
    args["exp_folder"] = "breed_vectors"
    args["out_exp_folder"] = local_exp_setup["folder_name"] + "/bv_eof_perturbations"

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    temp_processes, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    r_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def execute_pert_experiments(args: dict, exp_setup: dict):
    """Execute all perturbation experiments, i.e. run perturbations with different
    perturbation modes

    Parameters
    ----------
    args : dict
        Local run-time arguments
    exp_setup : dict
        Experiment setup
    """
    # Set licence
    cfg.LICENCE = exp.NORMAL_PERTURBATION

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["pert_setup"]

    # Prepare arguments for perturbation runs
    args["time_to_run"] = local_exp_setup["time_to_run"]
    args["endpoint"] = True
    args["n_profiles"] = local_exp_setup["n_units"]

    # Execute experiments
    bv_pert_experiment(copy.deepcopy(args), local_exp_setup)
    bv_eof_pert_experiment(copy.deepcopy(args), local_exp_setup)
    rd_pert_experiment(copy.deepcopy(args), local_exp_setup)
    nm_pert_experiment(copy.deepcopy(args), local_exp_setup)


def main(args: dict):
    """Main runner of the comparison

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    # Get experiment setup
    exp_file_path: pl.Path = pl.Path(
        "./params/experiment_setups/compare_pert_experiment_setups.json"
    )
    exp_setup: dict = exp_utils.get_exp_setup(exp_file_path, args)

    exp_utils.preprocess_exp_setup_for_comparison(exp_setup)

    # Update run-time arguments
    args["n_units"] = exp_setup["general"]["n_units"]

    # Generate perturbation vectors
    generate_vectors(copy.deepcopy(args), exp_setup)

    # Perform perturbation experiments
    execute_pert_experiments(copy.deepcopy(args), exp_setup)


if __name__ == "__main__":
    cfg.init_licence()
    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    args = mult_pert_arg_setup.args

    # Add ny argument
    if cfg.MODEL == Models.SHELL_MODEL:
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )

    g_ui.confirm_run_setup(args)

    main(args)
