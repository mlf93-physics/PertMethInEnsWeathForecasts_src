import sys

sys.path.append("..")
import copy
import pathlib as pl
import shell_model_experiments.params as sh_params
import shell_model_experiments.utils.util_funcs as sh_utils
import perturbation_runner as pt_runner
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
from general.params.env_params import *
from general.params.model_licences import Models
import general.utils.experiments.exp_utils as exp_utils
import general.utils.exceptions as g_exceptions
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
import config as cfg

# Get parameters for model
params = sh_params

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def main(args):
    exp_file_path = pl.Path(
        "./params/experiment_setups/hyper_diff_perturb_experiment_setups.json"
    )
    exp_setup = exp_utils.get_exp_setup(exp_file_path, args)
    processes = []

    # Prepare arguments
    args["out_exp_folder"] = exp_setup["folder_name"]
    args["start_times"] = exp_setup["start_times"]
    args["time_to_run"] = exp_setup["time_to_run"]
    args["endpoint"] = True
    args["n_profiles"] = len(exp_setup["start_times"])
    args["n_runs_per_profile"] = exp_setup["n_runs_per_profile"]
    args["diff_exponent"] = exp_setup["diff_exponent"]
    args["ny_n"] = exp_setup["ny_n"]

    # Prepare ny
    args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
        args["forcing"], args["ny_n"], args["diff_exponent"]
    )

    copy_args = copy.deepcopy(args)
    temp_processes, _, _ = pt_runner.main_setup(copy_args)
    processes.extend(temp_processes)

    if len(processes) > 0:
        pt_runner.main_run(
            processes,
            args=copy_args,
        )
    else:
        print("No processes to run - check if blocks already exists")

    # Save exp setup to exp folder
    g_save.save_exp_info(exp_setup, args)

    if args["erda_run"]:
        path = pl.Path(args["datapath"], exp_setup["folder_name"])
        g_save_utils.compress_dir(path, exp_setup["folder_name"])


if __name__ == "__main__":
    cfg.init_licence()

    # Check if correct model is running
    if cfg.MODEL != Models.SHELL_MODEL:
        g_exceptions.ModelError("Model is not valid for hyper diffusion experiments")
    # Get arguments
    pert_arg_setup = a_parsers.PerturbationArgSetup()
    pert_arg_setup.setup_parser()
    args = pert_arg_setup.args

    g_ui.confirm_run_setup(args)

    main(args)
