import sys

sys.path.append("..")
import argparse
import pathlib as pl
import copy
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import perturbation_runner as pt_runner
import general.utils.experiments.exp_utils as exp_utils
import general.utils.experiments.validate_exp_setups as ut_exp_val
import general.utils.runner_utils as r_utils
import general.utils.util_funcs as g_utils
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_breed_vector_funcs as br_save
import general.utils.exceptions as g_exceptions
import general.utils.perturb_utils as pt_utils
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def main(args):
    # Set exp_setup path
    exp_file_path = pl.Path(
        "./params/experiment_setups/breed_vector_experiment_setups.json"
    )
    # Get the current experiment setup
    exp_setup = exp_utils.get_exp_setup(exp_file_path, args)

    # Get number of existing blocks
    n_existing_units = g_utils.count_existing_files_or_dirs(
        search_path=pl.Path(args["path"], exp_setup["folder_name"]),
        search_pattern=".csv",
    )

    # Validate the start time method
    ut_exp_val.validate_start_time_method(exp_setup=exp_setup)

    # Generate start times
    start_times, num_possible_units = r_utils.generate_start_times(exp_setup, args)

    processes = []

    # Calculate the desired number of units
    for i in range(
        n_existing_units,
        min(args["num_units"] + n_existing_units, num_possible_units),
    ):

        # Make analysis forecasts
        args["time_to_run"] = exp_setup["time_per_cycle"]
        args["start_time"] = [start_times[i]]
        args["start_time_offset"] = exp_setup["vector_offset"]
        args["endpoint"] = True
        args["n_profiles"] = 1
        args["n_runs_per_profile"] = exp_setup["n_vectors"]
        args["perturb_folder"] = f"{exp_setup['folder_name']}"
        args = g_utils.adjust_start_times_with_offset(args)

        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        # Start off with None value in order to invoke random perturbations
        rescaled_data = None
        perturb_positions = None
        for _ in range(exp_setup["n_cycles"]):

            processes, data_out_list, perturb_positions = pt_runner.main_setup(
                copy_args,
                u_profiles_perturbed=rescaled_data,
                perturb_positions=perturb_positions,
                exp_setup=exp_setup,
            )
            print("processes", processes)

            if len(processes) > 0:
                # Run specified number of cycles
                pt_runner.main_run(
                    processes,
                    args=copy_args,
                    num_units=min(
                        copy_args["num_units"], num_possible_units - n_existing_units
                    ),
                )
                # Offset time to prepare for next run and import of reference data
                # for rescaling
                copy_args["start_time"][0] += copy_args["time_to_run"]

                # The rescaled data is used to start off cycle 1+
                exit()
                # rescaled_data = pt_utils.rescale_perturbations(data_out_list, copy_args)
        else:
            print("No processes to run - check if units already exists")

        # Save breed vector data
        br_save.save_breed_vector_unit(
            rescaled_data,
            perturb_position=perturb_positions,
            br_unit=i,
            args=args,
            exp_setup=exp_setup,
        )

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
