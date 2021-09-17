import sys

sys.path.append("..")
import copy
import argparse
import pathlib as pl
import json
import numpy as np
from shell_model_experiments.params.params import *
import shell_model_experiments.utils.validate_exp_setups as ut_val
import general.utils.util_funcs as g_utils
import general.utils.save_data_funcs as g_save
from shell_model_experiments.params.env_params import *
import perturbation_runner as pt_runner


def main(args):

    # Get experiment setup
    with open(
        "./params/experiment_setups/lorentz_block_experiment_setups.json", "r"
    ) as file:
        exp_setup_file = json.load(file)

    if args["exp_setup"] is None:
        raise ValueError("No experiment setup chosen")
        exit()
    else:
        exp_setup = exp_setup_file[args["exp_setup"]]

    # Validate setup
    ut_val.validate_lorentz_block_setup(exp_setup=exp_setup)

    # Get number of existing blocks
    n_existing_blocks = g_utils.count_existing_files_or_dirs(
        search_path=pl.Path(args["path"], exp_setup["folder_name"]), search_pattern="/"
    )

    processes = []

    for i in range(
        n_existing_blocks,
        min(args["num_blocks"] + n_existing_blocks, len(exp_setup["start_times"])),
    ):

        parent_perturb_folder = f"{exp_setup['folder_name']}/lorentz_block{i + 1}"

        # Make analysis forecasts
        args["time_to_run"] = exp_setup["time_to_run"]
        args["start_time"] = [exp_setup["start_times"][i]]
        args["start_time_offset"] = exp_setup["day_offset"]
        args["endpoint"] = True
        args["n_profiles"] = exp_setup["n_analyses"]
        args["n_runs_per_profile"] = 1
        args["perturb_folder"] = f"{parent_perturb_folder}/analysis_forecasts"

        args = g_utils.adjust_start_times_with_offset(args)

        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        processes.extend(pt_runner.main_setup(copy_args))

        # Make forecasts
        args["start_time"] = [exp_setup["start_times"][i] - exp_setup["day_offset"]]
        args["time_to_run"] = exp_setup["time_to_run"] + exp_setup["day_offset"]
        args["endpoint"] = True
        args["n_profiles"] = 1
        args["n_runs_per_profile"] = exp_setup["n_analyses"]
        args["perturb_folder"] = f"{parent_perturb_folder}/forecasts"

        copy_args = copy.deepcopy(args)
        processes.extend(pt_runner.main_setup(copy_args))

    if len(processes) > 0:
        pt_runner.main_run(
            processes,
            args=copy_args,
            num_blocks=min(
                args["num_blocks"], len(exp_setup["start_times"]) - n_existing_blocks
            ),
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
    arg_parser.add_argument("--n_runs_per_profile", default=1, type=int)
    arg_parser.add_argument("--n_profiles", default=1, type=int)
    arg_parser.add_argument("--start_time", nargs="+", type=float)
    arg_parser.add_argument("--eigen_perturb", action="store_true")
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    arg_parser.add_argument("--single_shell_perturb", default=None, type=int)
    arg_parser.add_argument("--num_blocks", default=np.inf, type=int)
    # arg_parser.add_argument("--block_step", default=0.2, type=float)
    arg_parser.add_argument("--start_time_offset", default=None, type=float)
    arg_parser.add_argument("--endpoint", action="store_true")
    arg_parser.add_argument("--erda_run", action="store_true")
    arg_parser.add_argument("--exp_setup", default=None, type=str)

    args = vars(arg_parser.parse_args())

    args["ref_run"] = False

    # Set seed if wished
    if args["seed_mode"]:
        np.random.seed(seed=1)

    main(args)

    # Find DONE sound to play
    done_file = pl.Path("/home/martin/Music/done_sound.mp3")

    # if os.path.isfile(done_file):
    #     playsound(done_file)
