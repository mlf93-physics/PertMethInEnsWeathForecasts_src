import os
import sys

sys.path.append("..")
import argparse
from pathlib import Path
import numpy as np
import multiprocessing
from pyinstrument import Profiler
from shell_model_experiments.sabra_model.sabra_model import run_model
from shell_model_experiments.params.params import *
from shell_model_experiments.utils.save_data_funcs import save_data, save_perturb_info
from shell_model_experiments.utils.import_data_funcs import import_start_u_profiles
from perturbation_runner import main as perturbation_main


def main(args):
    perturbation_main(args)


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", nargs="+", type=str)
    arg_parser.add_argument("--path", nargs="?", type=str)
    arg_parser.add_argument(
        "--perturb_folder", nargs="?", default=None, required=True, type=str
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
    arg_parser.add_argument("--start_time_offset", default=None, type=float)
    arg_parser.add_argument("--endpoint", action="store_true")

    args = vars(arg_parser.parse_args())

    args["ref_run"] = False

    if args["start_time"] is not None:
        if args["n_profiles"] > 1 and args["start_time_offset"] is None:
            np.testing.assert_equal(
                len(args["start_time"]),
                args["n_profiles"],
                "The number of start times do not equal the number of"
                + " requested profiles.",
            )
        elif args["n_profiles"] > 1 and args["start_time_offset"] is not None:
            np.testing.assert_equal(
                len(args["start_time"]), 1, "Too many start times given"
            )
            print(
                "Determining starttimes from single starttime value and the"
                + " start_time_offset parameter"
            )
            args["start_time"] = [
                args["start_time"][0] + args["start_time_offset"] * i
                for i in range(args["n_profiles"])
            ]
        else:
            np.testing.assert_equal(
                len(args["start_time"]), 1, "Too many start times given"
            )

    # Set seed if wished
    if args["seed_mode"]:
        np.random.seed(seed=1)

    main(args)
