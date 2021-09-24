import sys

sys.path.append("..")
import copy
import argparse
import pathlib as pl
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import perturbation_runner as pt_runner
import general.utils.util_funcs as g_utils
import general.utils.save_data_funcs as g_save
from general.params.env_params import *
from general.params.model_licences import Models
import general.utils.exceptions as g_exceptions
import general.utils.experiments.validate_exp_setups as ut_exp_val
import general.utils.experiments.exp_utils as exp_utils
import general.utils.runner_utils as r_utils
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def main(args):
    exp_file_path = pl.Path(
        "./params/experiment_setups/lorentz_block_experiment_setups.json"
    )
    exp_setup = exp_utils.get_exp_setup(exp_file_path, args)

    # Validate setup
    ut_exp_val.validate_lorentz_block_setup(exp_setup=exp_setup)

    # Get number of existing blocks
    n_existing_units = g_utils.count_existing_files_or_dirs(
        search_path=pl.Path(args["path"], exp_setup["folder_name"]), search_pattern="/"
    )

    # Determine how to infer start times (from start_times in exp_setup or
    # calculated from block_offset)
    ut_exp_val.validate_start_time_method(exp_setup=exp_setup)

    start_times, num_possible_units = r_utils.generate_start_times(exp_setup, args)

    processes = []

    for i in range(
        n_existing_units,
        min(args["num_units"] + n_existing_units, num_possible_units),
    ):

        parent_perturb_folder = f"{exp_setup['folder_name']}/lorentz_block{i}"

        # Make analysis forecasts
        args["time_to_run"] = exp_setup["time_to_run"]
        args["start_time"] = [start_times[i] + exp_setup["day_offset"]]
        args["start_time_offset"] = exp_setup["day_offset"]
        args["endpoint"] = True
        args["n_profiles"] = exp_setup["n_analyses"]
        args["n_runs_per_profile"] = 1
        args["perturb_folder"] = f"{parent_perturb_folder}/analysis_forecasts"

        args = g_utils.adjust_start_times_with_offset(args)

        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        temp_processes, _, _ = pt_runner.main_setup(copy_args)
        processes.extend(temp_processes)

        # Make forecasts
        args["start_time"] = [start_times[i]]
        args["time_to_run"] = exp_setup["time_to_run"] + exp_setup["day_offset"]
        args["endpoint"] = True
        args["n_profiles"] = 1
        args["n_runs_per_profile"] = exp_setup["n_analyses"]
        args["perturb_folder"] = f"{parent_perturb_folder}/forecasts"

        copy_args = copy.deepcopy(args)
        temp_processes, _, _ = pt_runner.main_setup(copy_args)
        processes.extend(temp_processes)

    if len(processes) > 0:
        pt_runner.main_run(
            processes,
            args=copy_args,
            num_units=min(args["num_units"], num_possible_units - n_existing_units),
        )
    else:
        print("No processes to run - check if blocks already exists")

    if args["erda_run"]:
        path = pl.Path(args["path"], exp_setup["folder_name"])
        g_save.compress_dir(path, "test_temp1")


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", nargs="+", type=str)
    arg_parser.add_argument("--path", nargs="?", type=str)
    arg_parser.add_argument(
        "--perturb_folder", nargs="?", default=None, required=False, type=str
    )
    arg_parser.add_argument("--time_to_run", default=0.1, type=float)
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--ny_n", default=None, type=int)
    arg_parser.add_argument("--sigma", default=10, type=float)
    arg_parser.add_argument("--r_const", default=28, type=float)
    arg_parser.add_argument("--b_const", default=8 / 3, type=float)
    arg_parser.add_argument("--n_runs_per_profile", default=1, type=int)
    arg_parser.add_argument("--n_profiles", default=1, type=int)
    arg_parser.add_argument("--start_time", nargs="+", type=float)
    arg_parser.add_argument("--pert_mode", default="random", type=str)
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    arg_parser.add_argument("--single_shell_perturb", default=None, type=int)
    arg_parser.add_argument("--num_units", default=np.inf, type=int)
    arg_parser.add_argument("--endpoint", action="store_true")
    arg_parser.add_argument("--erda_run", action="store_true")
    arg_parser.add_argument("--exp_setup", default=None, type=str)

    args = vars(arg_parser.parse_args())

    args["ref_run"] = False

    # Set seed if wished
    if args["seed_mode"]:
        np.random.seed(seed=1)

    # Check if arguments apply to the model at use
    if MODEL == Models.LORENTZ63:
        if args["single_shell_perturb"] is not None:
            raise g_exceptions.InvalidArgument(
                "single_shell_perturb is not a valid option for current model."
                + f" Model in use: {MODEL}",
                argument="single_shell_perturb",
            )
        elif args["ny_n"] is not None:
            raise g_exceptions.InvalidArgument(
                "ny_n is not a valid option for current model."
                + f" Model in use: {MODEL}",
                argument="ny_n",
            )

    main(args)

    # Find DONE sound to play
    done_file = pl.Path("/home/martin/Music/done_sound.mp3")

    # if os.path.isfile(done_file):
    #     playsound(done_file)
