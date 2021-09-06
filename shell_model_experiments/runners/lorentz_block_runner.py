import os
import sys

sys.path.append("..")
import argparse
from pathlib import Path
import numpy as np
from shell_model_experiments.sabra_model.sabra_model import run_model
from shell_model_experiments.params.params import *
from shell_model_experiments.utils.save_data_funcs import save_data, save_perturb_info
from shell_model_experiments.utils.import_data_funcs import import_start_u_profiles
from shell_model_experiments.utils.util_funcs import adjust_start_times_with_offset
from perturbation_runner import main as perturbation_main


def main(args):
    day_offset = 1 / 16
    time_to_run = 1.0

    for i in range(args["num_blocks"]):
        start_time_analysis1 = 9.5 + args["block_step"] * i
        parent_perturb_folder = f"test_lorentz_block_averaging/lorentz_block{i + 1}"
        # # Make analysis forecasts
        args["time_to_run"] = time_to_run
        args["start_time"] = [start_time_analysis1]
        args["start_time_offset"] = day_offset
        args["endpoint"] = True
        args["n_profiles"] = 8
        args["n_runs_per_profile"] = 1
        args["perturb_folder"] = f"{parent_perturb_folder}/analysis_forecasts"

        args = adjust_start_times_with_offset(args)

        perturbation_main(args)

        # Make forecasts
        args["start_time"] = [start_time_analysis1 - day_offset]
        args["time_to_run"] = time_to_run + day_offset
        args["endpoint"] = True
        args["n_profiles"] = 1
        args["n_runs_per_profile"] = 8
        args["perturb_folder"] = f"{parent_perturb_folder}/forecasts"
        perturbation_main(args)


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
    arg_parser.add_argument("--num_blocks", default=1, type=int)
    arg_parser.add_argument("--block_step", default=0.2, type=float)
    arg_parser.add_argument("--start_time_offset", default=None, type=float)
    arg_parser.add_argument("--endpoint", action="store_true")

    args = vars(arg_parser.parse_args())

    args["ref_run"] = False

    # Set seed if wished
    if args["seed_mode"]:
        np.random.seed(seed=1)

    main(args)

    # Find DONE sound to play
    done_file = Path("/home/martin/Music/done_sound.mp3")

    # if os.path.isfile(done_file):
    #     playsound(done_file)
