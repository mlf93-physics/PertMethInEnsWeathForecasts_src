import sys

sys.path.append("..")
import pathlib as pl
import copy
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import perturbation_runner as pt_runner
import general.utils.experiments.exp_utils as exp_utils
import general.utils.experiments.validate_exp_setups as ut_exp_val
import general.utils.runner_utils as r_utils
import general.utils.util_funcs as g_utils
import general.utils.saving.save_data_funcs as g_save
import general.utils.importing.import_data_funcs as g_import
import general.utils.saving.save_breed_vector_funcs as br_save
import general.utils.perturb_utils as pt_utils
import general.utils.argument_parsers as a_parsers
from general.params.model_licences import Models
from config import MODEL, GLOBAL_PARAMS

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params

# Set global params
GLOBAL_PARAMS.ref_run = False


def main(args):
    # Set exp_setup path
    exp_file_path = pl.Path(
        "./params/experiment_setups/lyapunov_vector_experiment_setups.json"
    )
    # Get the current experiment setup
    exp_setup = exp_utils.get_exp_setup(exp_file_path, args)

    # Get number of existing blocks
    n_existing_units = g_utils.count_existing_files_or_dirs(
        search_path=pl.Path(args["datapath"], exp_setup["folder_name"]),
        search_pattern="lyapunov_vector*.csv",
    )

    # Validate the start time method
    ut_exp_val.validate_start_time_method(exp_setup=exp_setup)

    # Generate start times
    start_times, num_possible_units = r_utils.generate_start_times(exp_setup, args)

    processes = []

    # Calculate the desired number of units
    for i in range(
        n_existing_units,
        min(args["n_units"] + n_existing_units, num_possible_units),
    ):

        # Make analysis forecasts
        args["time_to_run"] = exp_setup["integration_time"]
        args["start_times"] = [start_times[i]]
        args["start_time_offset"] = (
            exp_setup["vector_offset"] if "vector_offset" in exp_setup else None
        )
        args["endpoint"] = True
        args["n_profiles"] = 1
        args["n_runs_per_profile"] = exp_setup["n_vectors"]
        args["exp_folder"] = pl.Path(
            exp_setup["folder_name"], exp_setup["sub_exp_folder"]
        )
        args = g_utils.adjust_start_times_with_offset(args)

        # Prepare reference data import
        args["ref_start_time"] = start_times[i]
        args["ref_end_time"] = start_times[i] + args["time_to_run"]
        # Import reference data
        _, u_ref, ref_header_dict = g_import.import_ref_data(args=args)

        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        processes, _, _ = pt_runner.main_setup(
            copy_args, exp_setup=exp_setup, u_ref=u_ref
        )

        if len(processes) > 0:
            # Run specified number of cycles
            pt_runner.main_run(
                processes,
                args=copy_args,
            )

        else:
            print("No processes to run - check if units already exists")

    # Reset exp_folder
    args["exp_folder"] = exp_setup["folder_name"]
    # Save exp setup to exp folder
    g_save.save_exp_info(exp_setup, args)

    if args["erda_run"]:
        path = pl.Path(args["datapath"], exp_setup["folder_name"])
        g_save.compress_dir(path, "test_temp1")


if __name__ == "__main__":
    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    ref_arg_setup = a_parsers.ReferenceAnalysisArgParser()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    # Add submodel attribute
    MODEL.submodel = "TL"

    # Add ny argument
    if MODEL == Models.SHELL_MODEL:
        args["ny"] = params.ny_from_ny_n_and_forcing(args["forcing"], args["ny_n"])

    main(args)
