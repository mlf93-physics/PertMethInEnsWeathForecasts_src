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
import general.runners.perturbation_runner as pt_runner
import general.utils.argument_parsers as a_parsers
import general.utils.arg_utils as a_utils
import general.utils.experiments.exp_utils as exp_utils
import general.utils.process_utils as pr_utils
import general.utils.running.runner_utils as r_utils
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import general.utils.importing.import_data_funcs as g_import
from general.params.experiment_licences import Experiments as exp
from general.params.model_licences import Models
from general.runners.lyapunov_vector_runner import main as lv_runner
from general.runners.breed_vector_runner import main as bv_runner
from general.analyses.breed_vector_eof_analysis import main as bv_eof_analyser
from general.runners.singular_vector_lanczos_runner import main as sv_runner
from general.runners.tangent_linear_runner import main as tl_runner
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
    local_exp_setup["folder_name"] = str(
        pl.Path(
            local_exp_setup["folder_name"],
            local_exp_setup["vector_folder"],
            "lv_vectors",
        )
    )

    lv_runner(args, local_exp_setup)

    # Reset submodel
    cfg.MODEL.submodel = None


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
    local_exp_setup["folder_name"] = str(
        pl.Path(
            local_exp_setup["folder_name"],
            local_exp_setup["vector_folder"],
            "bv_vectors",
        )
    )

    bv_runner(args, local_exp_setup)


def generate_bv_eofs(args: dict, exp_setup: dict):
    """Generate the BV-EOFs according to the exp setup

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """
    print(f"{col.Fore.GREEN}BV-EOF GENERATION{col.Fore.RESET}")
    # Update licence
    cfg.LICENCE = exp.BREEDING_EOF_VECTORS

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["bv_eof_gen_setup"]
    # Set local args params
    args["out_exp_folder"] = pl.Path(
        local_exp_setup["folder_name"],
        local_exp_setup["vector_folder"],
        "bv_eof_vectors",
    )
    args["n_profiles"] = local_exp_setup["n_units"]
    args["n_runs_per_profile"] = local_exp_setup["n_vectors"]
    args["pert_vector_folder"] = pl.Path(
        local_exp_setup["folder_name"], local_exp_setup["vector_folder"]
    )
    args["exp_folder"] = "bv_vectors"

    bv_eof_analyser(args, exp_setup=local_exp_setup)


def generate_svs(args: dict, exp_setup: dict):
    """Generate the SVs according to the exp setup

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """
    print(f"{col.Fore.GREEN}SV GENERATION{col.Fore.RESET}")

    # Update licence
    cfg.LICENCE = exp.SINGULAR_VECTORS

    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["sv_gen_setup"]
    # Set local exp values
    local_exp_setup["folder_name"] = str(
        pl.Path(
            local_exp_setup["folder_name"],
            local_exp_setup["vector_folder"],
            "sv_vectors",
        )
    )

    sv_runner(args, exp_setup=local_exp_setup)


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
    args["save_no_pert"] = True

    if "bv" in args["vectors"] or "all" in args["vectors"]:
        generate_bvs(copy.deepcopy(args), copy.deepcopy(exp_setup))
    if "lv" in args["vectors"] or "all" in args["vectors"]:
        generate_lvs(copy.deepcopy(args), copy.deepcopy(exp_setup))
    if "bv_eof" in args["vectors"] or "all" in args["vectors"]:
        generate_bv_eofs(copy.deepcopy(args), copy.deepcopy(exp_setup))
    if "sv" in args["vectors"] or "all" in args["vectors"]:
        generate_svs(copy.deepcopy(args), copy.deepcopy(exp_setup))


def rd_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with rd as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """
    print(f"{col.Fore.GREEN}RD PERTURBATIONS{col.Fore.RESET}")

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = local_exp_setup["n_runs_per_profile"]
    args["pert_mode"] = "rd"
    args["out_exp_folder"] = pl.Path(
        local_exp_setup["folder_name"],
        "rd_perturbations",
    )

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    temp_processes, _, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    pr_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def tl_rd_pert_experiment(args: dict, local_exp_setup: dict):
    """Run TL perturbations with rd as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """
    print(f"{col.Fore.GREEN}TL with RD PERTURBATIONS{col.Fore.RESET}")

    # Update licence
    cfg.LICENCE = exp.TANGENT_LINEAR

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = local_exp_setup["n_runs_per_profile"]
    args["pert_mode"] = "rd"
    args["out_exp_folder"] = pl.Path(
        local_exp_setup["folder_name"],
        "tl_rd_perturbations",
    )

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    tl_runner(copy_args, exp_setup=local_exp_setup)


def lv_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with lv as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """
    print(f"{col.Fore.GREEN}LV PERTURBATIONS{col.Fore.RESET}")

    processes = []

    # Prepare arguments for perturbation run
    args[
        "n_runs_per_profile"
    ] = 1  # In order to separate LV0, LV1, LV2, ... out into separate folders
    args["pert_vector_folder"] = pl.Path(
        local_exp_setup["folder_name"], local_exp_setup["vector_folder"]
    )
    args["pert_mode"] = "lv"
    args["exp_folder"] = "lv_vectors"
    args["out_exp_folder"] = pl.Path(
        local_exp_setup["folder_name"],
        "lv_perturbations",
    )

    for i in range(local_exp_setup["n_vectors"]):
        args["specific_start_vector"] = i
        # Prepare vector str index
        str_index = lib_type_utils.zpad_string(str(i), n_zeros=2)
        args["out_exp_folder"] = pl.Path(
            local_exp_setup["folder_name"],
            f"lv{str_index}_perturbations",
        )
        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        processes, _, _, _ = pt_runner.main_setup(copy_args)

        pr_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def nm_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with nm as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """
    print(f"{col.Fore.GREEN}NM PERTURBATIONS{col.Fore.RESET}")

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = local_exp_setup["n_runs_per_profile"]
    args["n_profiles"] = local_exp_setup["n_units"]
    args["pert_mode"] = "nm"
    args["out_exp_folder"] = pl.Path(
        local_exp_setup["folder_name"],
        "nm_perturbations",
    )

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    temp_processes, _, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    pr_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def bv_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with BVs as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """
    print(f"{col.Fore.GREEN}BV PERTURBATIONS{col.Fore.RESET}")

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = local_exp_setup["n_runs_per_profile"]
    args["pert_mode"] = "bv"
    args["pert_vector_folder"] = pl.Path(
        local_exp_setup["folder_name"], local_exp_setup["vector_folder"]
    )
    args["exp_folder"] = "bv_vectors"
    args["out_exp_folder"] = pl.Path(
        local_exp_setup["folder_name"],
        "bv_perturbations",
    )

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    temp_processes, _, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    pr_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def bv_eof_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with BV-EOFs as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """
    print(f"{col.Fore.GREEN}BV-EOF PERTURBATIONS{col.Fore.RESET}")

    # Prepare arguments for perturbation run
    args[
        "n_runs_per_profile"
    ] = 1  # In order to separate BV-EOF0, BV-EOF1, BV-EOF2, ... out into separate folders
    args["pert_mode"] = "bv_eof"
    args["pert_vector_folder"] = pl.Path(
        local_exp_setup["folder_name"], local_exp_setup["vector_folder"]
    )
    args["exp_folder"] = "bv_eof_vectors"

    for i in range(local_exp_setup["n_vectors"]):
        args["specific_start_vector"] = i
        # Prepare vector str index
        str_index = lib_type_utils.zpad_string(str(i), n_zeros=2)
        args["out_exp_folder"] = pl.Path(
            local_exp_setup["folder_name"],
            f"bv_eof{str_index}_perturbations",
        )
        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        processes, _, _, _ = pt_runner.main_setup(copy_args)

        pr_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def sv_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with SV as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """
    print(f"{col.Fore.GREEN}SV PERTURBATIONS{col.Fore.RESET}")

    # Prepare arguments for perturbation run
    args[
        "n_runs_per_profile"
    ] = 1  # In order to separate SV0, SV1, SV2, ... out into separate folders
    args["pert_mode"] = "sv"
    args["pert_vector_folder"] = pl.Path(
        local_exp_setup["folder_name"], local_exp_setup["vector_folder"]
    )
    args["exp_folder"] = "sv_vectors"

    for i in range(local_exp_setup["n_vectors"]):
        args["specific_start_vector"] = i
        # Prepare vector str index
        str_index = lib_type_utils.zpad_string(str(i), n_zeros=2)
        args["out_exp_folder"] = pl.Path(
            local_exp_setup["folder_name"],
            f"sv{str_index}_perturbations",
        )
        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        processes, _, _, _ = pt_runner.main_setup(copy_args)

        pr_utils.run_pert_processes(copy_args, local_exp_setup, processes)


def rf_pert_experiment(args: dict, local_exp_setup: dict):
    """Run perturbations with SV as pert_mode

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    """
    print(f"{col.Fore.GREEN}RF PERTURBATIONS{col.Fore.RESET}")

    processes = []

    # Prepare arguments for perturbation run
    args["n_runs_per_profile"] = local_exp_setup["n_runs_per_profile"]
    args["n_profiles"] = local_exp_setup["n_units"]
    args["pert_mode"] = "rf"
    args["out_exp_folder"] = pl.Path(
        local_exp_setup["folder_name"],
        "rf_perturbations",
    )

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)

    temp_processes, _, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    pr_utils.run_pert_processes(copy_args, local_exp_setup, processes)


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

    # Only generate start times if not requesting regime start
    if args["regime_start"] is None:
        # Generate start times
        if "eval_times" in local_exp_setup and "unit_offset" in local_exp_setup:
            args["start_times"] = local_exp_setup["eval_times"]
            args["start_time_offset"] = local_exp_setup["unit_offset"]
            args = g_utils.adjust_start_times_with_offset(args)
        else:
            # Check if ref path exists
            ref_file_path = pl.Path(args["datapath"], "ref_data")
            # Get ref info text file
            ref_header_dict = g_import.import_info_file(ref_file_path)
            args["start_times"] = r_utils.get_random_start_times(
                args, args["n_profiles"], ref_header_dict
            )

    elif cfg.MODEL == Models.SHELL_MODEL:
        start_times, num_possible_units, _ = sh_r_utils.get_regime_start_times(args)
        args["start_times"] = start_times[: args["n_profiles"]]

    # Execute experiments

    if "bv" in args["perturbations"] or "all" in args["perturbations"]:
        bv_pert_experiment(
            copy.deepcopy(args), local_exp_setup | exp_setup["bv_pert_setup"]
        )
    if "bv_eof" in args["perturbations"] or "all" in args["perturbations"]:
        bv_eof_pert_experiment(
            copy.deepcopy(args), local_exp_setup | exp_setup["bv_eof_gen_setup"]
        )
    if "lv" in args["perturbations"] or "all" in args["perturbations"]:
        lv_pert_experiment(
            copy.deepcopy(args), local_exp_setup | exp_setup["lv_gen_setup"]
        )
    if "rd" in args["perturbations"] or "all" in args["perturbations"]:
        rd_pert_experiment(
            copy.deepcopy(args), local_exp_setup | exp_setup["rd_pert_setup"]
        )
    if "tl_rd" in args["perturbations"] or "all" in args["perturbations"]:
        tl_rd_pert_experiment(
            copy.deepcopy(args), local_exp_setup | exp_setup["tl_rd_pert_setup"]
        )
    if "nm" in args["perturbations"] or "all" in args["perturbations"]:
        nm_pert_experiment(
            copy.deepcopy(args), local_exp_setup | exp_setup["nm_pert_setup"]
        )
    if "sv" in args["perturbations"] or "all" in args["perturbations"]:
        sv_pert_experiment(
            copy.deepcopy(args), local_exp_setup | exp_setup["sv_gen_setup"]
        )
    if "rf" in args["perturbations"] or "all" in args["perturbations"]:
        rf_pert_experiment(
            copy.deepcopy(args), local_exp_setup | exp_setup["rf_pert_setup"]
        )


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

    if len(args["vectors"]) > 0:
        # Generate perturbation vectors
        generate_vectors(copy.deepcopy(args), exp_setup)

    if len(args["perturbations"]) > 0:
        # Perform perturbation experiments
        execute_pert_experiments(copy.deepcopy(args), exp_setup)


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
