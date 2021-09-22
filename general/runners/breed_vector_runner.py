import sys

sys.path.append("..")
import argparse
import pathlib as pl
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
from general.params.model_licences import Models
import general.utils.experiments.exp_utils as exp_utils
import general.utils.experiments.validate_exp_setups as ut_exp_val
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

    ut_exp_val.validate_start_time_method(exp_setup=exp_setup)

    # start_times, num_possible_blocks = r_utils.generate_start_times(exp_setup, args)

    # Make analysis forecasts
    args["time_to_run"] = exp_setup["time_to_run"]
    args["start_time"] = [start_times[i]]
    args["start_time_offset"] = exp_setup["day_offset"]
    args["endpoint"] = True
    args["n_profiles"] = 1
    args["n_runs_per_profile"] = exp_setup["n_vectors"]
    args["perturb_folder"] = f"{exp_setup['folder_name']}"

    args = g_utils.adjust_start_times_with_offset(args)

    # Copy args in order not override in forecast processes
    copy_args = copy.deepcopy(args)


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
    arg_parser.add_argument("--num_blocks", default=np.inf, type=int)
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
