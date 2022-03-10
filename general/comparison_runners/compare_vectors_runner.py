"""Runs a complete set of perturbations based on several different perturbation
methods [rd, bv, bv-eof, ...]. Used to compare the different perturbation methods

Example
-------
python ../general/comparison_runners/compare_perturbations_runner.py --exp_setup=TestRun0

"""

import sys

sys.path.append("..")
import copy
import pathlib as pl
import colorama as col
import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.arg_utils as a_utils
import general.utils.experiments.exp_utils as exp_utils
import general.utils.running.runner_utils as r_utils
import general.utils.user_interface as g_ui
from general.params.experiment_licences import Experiments as exp
from general.params.model_licences import Models
from general.runners.breed_vector_runner import main as bv_runner
from general.runners.lyapunov_vector_runner import main as lv_runner
from general.runners.adj_lyapunov_vector_runner import main as adj_lv_runner
from general.runners.singular_vector_lanczos_runner import main as sv_runner
from ku_project.general.runners.final_singular_vector_lanczos_runner import (
    main as fsv_runner,
)
from libs.libutils import type_utils as lib_type_utils

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.util_funcs as sh_utils
    import shell_model_experiments.utils.runner_utils as sh_r_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params

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
    print(f"{col.Fore.GREEN}BV GENERATION{col.Fore.RESET}")
    # Update licence
    cfg.LICENCE = exp.BREEDING_VECTORS

    # Set local args params
    args["pert_mode"] = "rd"

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["bv_gen_setup"]

    for iw in local_exp_setup["iws"]:
        iw_str: str = lib_type_utils.zpad_string(str(iw), n_zeros=3)
        # Set local exp values
        local_exp_setup["folder_name"] = str(
            pl.Path(
                exp_setup["general"]["folder_name"],
                exp_setup["general"]["vector_folder"],
                f"bv_vectors_iw{iw_str}",
            )
        )
        local_exp_setup["integration_time"] = iw * params.dt

        bv_runner(args, local_exp_setup)


def generate_lvs(args: dict, exp_setup: dict):
    """Generate the LVs according to the exp setup

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """
    print(f"{col.Fore.GREEN}LV GENERATION{col.Fore.RESET}")
    # Update licence
    cfg.LICENCE = exp.LYAPUNOV_VECTORS

    # Set local args params
    args["pert_mode"] = "rd"

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["lv_gen_setup"]

    stored_eval_times = copy.copy(local_exp_setup["eval_times"])

    # Calculate lv valid at eval time + all end times
    for iw in [0]:  # *local_exp_setup["iws"]]:
        iw_str: str = lib_type_utils.zpad_string(str(iw), n_zeros=3)
        local_exp_setup["folder_name"] = str(
            pl.Path(
                exp_setup["general"]["folder_name"],
                exp_setup["general"]["vector_folder"],
                f"lv_vectors_val{iw_str}",
            )
        )

        local_exp_setup["eval_times"][0] = stored_eval_times[0] + iw * params.dt

        lv_runner(args, exp_setup=local_exp_setup)


def generate_adj_lvs(args: dict, exp_setup: dict):
    """Generate the adjoint LVs according to the exp setup

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """
    print(f"{col.Fore.GREEN}ADJ LV GENERATION{col.Fore.RESET}")
    # Update licence
    cfg.LICENCE = exp.ADJ_LYAPUNOV_VECTORS

    # Set local args params
    args["pert_mode"] = "rd"

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["lv_gen_setup"]

    stored_eval_times = copy.copy(local_exp_setup["eval_times"])

    # Calculate lv valid at eval time + all end times
    for iw in [0, *local_exp_setup["iws"]]:
        iw_str: str = lib_type_utils.zpad_string(str(iw), n_zeros=3)
        local_exp_setup["folder_name"] = str(
            pl.Path(
                exp_setup["general"]["folder_name"],
                exp_setup["general"]["vector_folder"],
                f"alv_vectors_val{iw_str}",
            )
        )

        local_exp_setup["eval_times"][0] = stored_eval_times[0] + iw * params.dt

        adj_lv_runner(args, exp_setup=local_exp_setup)


def generate_initial_svs(args: dict, exp_setup: dict):
    """Generate the initial SVs according to the exp setup

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """
    print(f"{col.Fore.GREEN}INITIAL TIME SV GENERATION{col.Fore.RESET}")

    # Update licence
    cfg.LICENCE = exp.SINGULAR_VECTORS

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["sv_gen_setup"]

    for iw in local_exp_setup["iws"]:
        iw_str: str = lib_type_utils.zpad_string(str(iw), n_zeros=3)
        # Set local exp values
        local_exp_setup["folder_name"] = str(
            pl.Path(
                exp_setup["general"]["folder_name"],
                exp_setup["general"]["vector_folder"],
                f"sv_vectors_iw{iw_str}",
            )
        )
        local_exp_setup["integration_time"] = iw * params.dt

        sv_runner(args, exp_setup=local_exp_setup)


def generate_final_svs(args: dict, exp_setup: dict):
    """Generate the final SVs according to the exp setup

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """

    print(f"{col.Fore.GREEN}FINAL TIME SV GENERATION{col.Fore.RESET}")

    # Update licence
    cfg.LICENCE = exp.FINAL_SINGULAR_VECTORS

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["fsv_gen_setup"]
    # Set pert_vector_folder to be able to get sv vectors for perturbations
    args["pert_vector_folder"] = pl.Path(
        exp_setup["general"]["folder_name"],
        exp_setup["general"]["vector_folder"],
    )

    stored_eval_times = copy.copy(local_exp_setup["eval_times"])

    # Set local args
    args["n_profiles"] = local_exp_setup["n_units"]

    for iw in local_exp_setup["iws"]:
        iw_str: str = lib_type_utils.zpad_string(str(iw), n_zeros=3)
        # Set local exp values
        local_exp_setup["folder_name"] = str(
            pl.Path(
                exp_setup["general"]["folder_name"],
                exp_setup["general"]["vector_folder"],
                f"fsv_vectors_iw{iw_str}",
            )
        )
        local_exp_setup["integration_time"] = iw * params.dt
        # Update eval_time
        local_exp_setup["eval_times"][0] = stored_eval_times[0] + iw * params.dt
        # Set exp_folder to get relevant sv vectors
        args["exp_folder"] = f"fsv_vectors_iw{iw_str}"

        fsv_runner(args, exp_setup=local_exp_setup)


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

    if "bv" in args["vectors"] or "all" in args["vectors"]:
        generate_bvs(copy.deepcopy(args), copy.deepcopy(exp_setup))
    if "lv" in args["vectors"] or "all" in args["vectors"]:
        generate_lvs(copy.deepcopy(args), copy.deepcopy(exp_setup))
    if "alv" in args["vectors"] or "all" in args["vectors"]:
        generate_adj_lvs(copy.deepcopy(args), copy.deepcopy(exp_setup))
    if "sv" in args["vectors"] or "all" in args["vectors"]:
        generate_initial_svs(copy.deepcopy(args), copy.deepcopy(exp_setup))
    if "fsv" in args["vectors"] or "all" in args["vectors"]:
        generate_final_svs(copy.deepcopy(args), copy.deepcopy(exp_setup))


def main(args: dict):
    """Main runner of the comparison

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    # Get experiment setup
    exp_file_path: pl.Path = pl.Path(
        "./params/experiment_setups/compare_vector_experiment_setups.json"
    )
    exp_setup: dict = exp_utils.get_exp_setup(exp_file_path, args)

    exp_utils.preprocess_exp_setup_for_comparison(exp_setup)

    # Update run-time arguments
    args["n_units"] = exp_setup["general"]["n_units"]

    if len(args["vectors"]) > 0:
        # Generate perturbation vectors
        generate_vectors(copy.deepcopy(args), exp_setup)


if __name__ == "__main__":
    cfg.init_licence()
    # Get arguments
    _parser = a_parsers.ComparisonArgParser()
    _parser.setup_parser()
    args = _parser.args

    a_utils.react_on_comparison_arguments(args)

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)
        # Add ny argument
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )

    g_ui.confirm_run_setup(args)
    r_utils.adjust_run_setup(args)

    main(args)
